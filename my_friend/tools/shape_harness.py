"""샌드박스 subprocess 안에서 실행되는 검증 하네스. torch + stdlib만 의존 (my_friend import 금지).

    python shape_harness.py <config.json>   →  config["result_path"]에 JSON 리포트 기록

단계:
  1. import / build
  2. meta pass   : torch.device("meta") 위 forward — 연산 비용 0으로 shape 체인 추적.
                   nonzero/.item() 등 data-dependent op, meta kernel이 없는 op(unique/masked_select)은
                   meta에서 실패 → 실측으로 폴백. 원인(종류/op/모듈/코드 줄)은 meta_detail에 기록.
  3. real pass   : torch.randn 더미 입력으로 forward + backward (NaN/Inf, unused param, activation 메모리)
  4. shape check : planner 명세의 module output shape과 실측 비교
  5. leakage     : causal 축의 t0 이후 입력을 교란 → 출력[:t0]이 변하면 미래 정보 누수
  6. lightning   : (LightningModule 파일이 있으면) Trainer(fast_dev_run=1)로 train/val 1 step —
                   생성, configure_optimizers, 유한한 loss, 옵티마이저가 파라미터를 실제로 갱신했는지
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import torch
from torch import nn

_DTYPES = {
    "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
    "int64": torch.int64, "bool": torch.bool,
}
MAX_SHAPE_LOG = 300


def _make_inputs(specs: List[dict], device: str, seed: int = 0) -> List[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for s in specs:
        dt = _DTYPES[s.get("dtype", "float32")]
        shape = tuple(s["shape"])
        if dt == torch.int64:
            # 토큰 id류: vocab 크기를 모르므로 보수적으로 [0, 2) — 임베딩 out-of-range를 피한다
            t = torch.randint(0, s.get("high", 2), shape, generator=g)
        elif dt == torch.bool:
            t = torch.ones(shape, dtype=torch.bool)
        else:
            t = torch.randn(shape, generator=g).to(dt)
        out.append(t.to(device))
    return out


def _shape_of(x: Any) -> Optional[str]:
    if isinstance(x, torch.Tensor):
        return "x".join(str(d) for d in x.shape) or "scalar"
    if isinstance(x, (tuple, list)):
        parts = [_shape_of(v) for v in x]
        return "(" + ", ".join(p for p in parts if p) + ")" if any(parts) else None
    if isinstance(x, dict):
        parts = {k: _shape_of(v) for k, v in x.items()}
        return "{" + ", ".join(f"{k}: {v}" for k, v in parts.items() if v) + "}"
    return None


def _tensors(x: Any) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for v in x for t in _tensors(v)]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _tensors(v)]
    return []


def _primary(out: Any) -> torch.Tensor:
    if isinstance(out, dict) and isinstance(out.get("loss"), torch.Tensor):
        return out["loss"]
    ts = [t for t in _tensors(out) if t.is_floating_point()]
    if not ts:
        raise TypeError(f"forward() returned no floating-point tensor: {_shape_of(out)}")
    return ts[0]


class _Recorder:
    """모든 서브모듈에 forward hook → (module path, output shape, bytes)."""

    def __init__(self, model: nn.Module) -> None:
        self.records: List[Dict[str, Any]] = []
        self.act_bytes = 0
        self._handles = [
            m.register_forward_hook(self._hook(name or "<root>")) for name, m in model.named_modules()
        ]

    def _hook(self, name: str):
        def fn(_m, _inp, out):
            for t in _tensors(out):
                if t.device.type != "meta":
                    self.act_bytes += t.numel() * t.element_size()
            if len(self.records) < MAX_SHAPE_LOG:
                self.records.append({"module": name, "output_shape": _shape_of(out) or "?"})
        return fn

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def _load_class(cfg: dict):
    sys.path.insert(0, cfg["workdir"])
    path = f"{cfg['workdir']}/{cfg['entry_file']}"
    mod_name = cfg["entry_file"][:-3].replace("/", ".")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, cfg["entry_class"])


# meta 텐서의 "값"을 읽으려 할 때 나는 메시지 (torch 2.14 기준 실측)
_DATA_DEP_MARKERS = (
    "item() cannot be called on meta", "cannot copy out of meta", "data-dependent", "no data!",
    "raises unimplemented by default",          # nonzero: 출력 크기가 값에 의존
    "without output_size",                       # repeat_interleave(tensor repeats)
    "can't convert meta device type tensor",     # .numpy()
)


def _classify_meta_error(exc: BaseException) -> tuple:
    """(kind, op). kind: data_dependent | unsupported_op | shape_error | error.
    shape_error는 meta에서도 잡히는 진짜 shape 버그 — real pass도 같은 곳에서 실패할 가능성이 높다."""
    msg = str(exc)
    low = msg.lower()
    m = re.search(r"\baten::([\w.]+)", msg) or re.search(r"\btorch\.([\w.]+)\(\)", msg) \
        or re.search(r"\bTensor\.(\w+)\(\)", msg)
    op = m.group(1) if m else None
    if any(k.lower() in low for k in _DATA_DEP_MARKERS):
        return "data_dependent", op
    if "no fake impl or meta kernel registered" in low or ("could not run" in low and "'meta' backend" in low):
        return "unsupported_op", op
    if isinstance(exc, RuntimeError):
        return "shape_error", op
    return "error", op


def _user_frame(tb, workdir: str) -> Optional[str]:
    """traceback에서 생성 코드(workdir 안 파일)의 가장 안쪽 프레임 → 'model.py:42  idx = torch.nonzero(m)'."""
    hit = None
    for fs in traceback.extract_tb(tb):
        if fs.filename.startswith(workdir.rstrip("/") + "/"):
            hit = f"{fs.filename[len(workdir.rstrip('/')) + 1:]}:{fs.lineno}  {(fs.line or '').strip()}"
    return hit


def _meta_pass(cfg: dict, cls) -> tuple:
    """torch.device('meta')에서 build + forward. 실패하면 (kind, detail)로 원인을 구조화해 돌려준다.
    module은 실패 시점에 forward 중이던 가장 안쪽 서브모듈 (pre-hook push / hook pop 스택)."""
    stage, active, handles = "build", [], []

    def push(path: str):
        def fn(_m, _inp):
            active.append(path)                   # 반환값 None 필수: hook이 값을 돌려주면 입력/출력이 바뀐다
        return fn

    def pop(_m, _inp, _out):
        if active:
            active.pop()
    try:
        with torch.device("meta"):
            meta_model = cls(**cfg.get("init_kwargs", {}))
        stage = "forward"
        for name, m in meta_model.named_modules():
            handles.append(m.register_forward_pre_hook(push(name or "<root>")))
            handles.append(m.register_forward_hook(pop))
        meta_in = _make_inputs(cfg["inputs"], "meta")
        with torch.no_grad():
            meta_model(*meta_in)
        return "ok", None
    except Exception as e:  # noqa: BLE001 — meta 실패는 리포트용 정보일 뿐
        kind, op = _classify_meta_error(e)
        return kind, {"kind": kind, "stage": stage, "op": op, "module": active[-1] if active else None,
                      "location": _user_frame(e.__traceback__, cfg["workdir"]),
                      "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        for h in handles:
            h.remove()


def run(cfg: dict) -> dict:
    rep: Dict[str, Any] = {"status": "pass", "failed_stage": None, "error": None, "traceback": None}
    t_start = time.time()
    stage = "import"

    def fail(st: str, exc: BaseException) -> dict:
        rep.update(status="fail", failed_stage=st, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(limit=8))
        rep["duration_s"] = round(time.time() - t_start, 3)
        return rep

    try:
        cls = _load_class(cfg)
        stage = "build"
        torch.manual_seed(0)
        model: nn.Module = cls(**cfg.get("init_kwargs", {}))
    except Exception as e:  # noqa: BLE001 — 하네스는 모든 예외를 리포트로 변환
        return fail(stage, e)

    n_params = sum(p.numel() for p in model.parameters())
    rep["param_count"] = n_params
    rep["param_mb"] = round(sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20, 3)
    # Adam: weight + grad + exp_avg + exp_avg_sq (fp32) ≈ 16B/param
    rep["est_adam_state_mb"] = round(n_params * 16 / 2**20, 3)

    # ---- 2. meta pass: 실패해도 판정에는 쓰지 않는다 (real pass가 기준) — 원인만 meta_detail에 남긴다
    if cfg.get("meta_pass", True):
        rep["meta_pass"], rep["meta_detail"] = _meta_pass(cfg, cls)

    # ---- 3. real pass
    device = cfg.get("device", "cpu")
    model.to(device).train()
    inputs = _make_inputs(cfg["inputs"], device)
    rec = _Recorder(model)
    try:
        stage = "forward"
        out = model(*inputs)
        rec.close()
        rep["shape_log"] = rec.records
        rep["activation_mb"] = round(rec.act_bytes / 2**20, 3)
        b, train_b = cfg.get("batch_size") or 1, cfg.get("train_batch_size") or 1
        # activation은 배치에 선형 (seq/spatial 크기는 검증값 기준임에 주의 — 리포트에서 명시)
        rep["est_train_activation_mb"] = round(rep["activation_mb"] * train_b / b, 3)

        primary = _primary(out)
        rep["nan_detected"] = bool(any(not torch.isfinite(t).all() for t in _tensors(out) if t.is_floating_point()))

        stage = "backward"
        primary.float().mean().backward()
        rep["unused_params"] = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None][:50]
    except Exception as e:  # noqa: BLE001
        rec.close()
        rep["shape_log"] = rec.records          # 실패 직전까지의 shape 체인이 디버깅의 핵심 단서
        return fail(stage, e)

    # ---- 4. shape check vs spec
    observed = {}
    for r in rep["shape_log"]:
        observed.setdefault(r["module"], r["output_shape"])   # 첫 호출 기준
    mismatches = []
    for exp in cfg.get("expected", []):
        want = "x".join(str(d) for d in exp["shape"])
        got = observed.get(exp["module"])
        if got is None:
            mismatches.append(f"{exp['module']}: module not found (expected {want})")
        elif got != want:
            mismatches.append(f"{exp['module']}: expected {want}, got {got}")
    for r in rep["shape_log"]:
        e = next((x for x in cfg.get("expected", []) if x["module"] == r["module"]), None)
        if e is not None:
            r["expected"] = "x".join(str(d) for d in e["shape"])
            r["ok"] = r["expected"] == r["output_shape"]
    out_exp = cfg.get("output_expected")
    if out_exp:
        got = [list(t.shape) for t in _tensors(out)]
        if got[: len(out_exp)] != out_exp:
            mismatches.append(f"forward output: expected {out_exp}, got {got}")
    rep["shape_mismatches"] = mismatches

    # ---- 5. causal leakage probe
    causal = cfg.get("causal")
    rep["leakage"] = {"checked": False, "leaked": False, "max_abs_diff": 0.0, "detail": ""}
    if causal:
        try:
            model.eval()
            with torch.no_grad():
                y1 = _primary(model(*inputs))
                perturbed = []
                lengths = []
                for x, axis in zip(inputs, causal["input_axes"]):
                    if axis is None or x.dtype == torch.bool:
                        perturbed.append(x)
                        continue
                    T = x.shape[axis]
                    lengths.append(T)
                    t0 = T // 2
                    x2 = x.clone()
                    future = x2.narrow(axis, t0 + 1, T - t0 - 1)    # view → in-place 수정이 x2에 반영
                    if x.is_floating_point():
                        future.copy_(torch.randn_like(future) * 3)
                    else:
                        future.copy_(1 - future)                      # 토큰 id {0,1} 반전 → 반드시 변화
                    perturbed.append(x2)
                if lengths:
                    t0 = min(lengths) // 2
                    y2 = _primary(model(*perturbed))
                    oa = causal["output_axis"]
                    diff = (y1.narrow(oa, 0, t0 + 1) - y2.narrow(oa, 0, t0 + 1)).abs().max().item()
                    rep["leakage"] = {
                        "checked": True, "leaked": diff > 1e-5, "max_abs_diff": diff,
                        "detail": f"perturbed inputs at t>{t0} along causal axis; compared outputs[:{t0 + 1}]",
                    }
        except Exception as e:  # noqa: BLE001
            rep["leakage"] = {"checked": False, "leaked": False, "max_abs_diff": 0.0,
                              "detail": f"probe failed: {type(e).__name__}: {e}"}

    # ---- 6. lightning: 앞 단계가 모두 통과한 모델에 대해서만 (실패 원인을 섞지 않기 위해)
    rep["lightning"] = {"checked": False}
    if cfg.get("lightning") and not mismatches and not rep["nan_detected"] and not rep["leakage"]["leaked"]:
        try:
            rep["lightning"] = check_lightning(cfg, cls)
        except Exception as e:  # noqa: BLE001 — import/클래스 탐색 실패
            rep["lightning"] = {"checked": True, "file": cfg["lightning"]["file"], "error": f"{type(e).__name__}: {e}",
                                "traceback": traceback.format_exc(limit=8)}

    if mismatches:
        rep.update(status="fail", failed_stage="shape", error="; ".join(mismatches[:5]))
    elif rep["nan_detected"]:
        rep.update(status="fail", failed_stage="numerical", error="NaN/Inf in forward output")
    elif rep["leakage"]["leaked"]:
        rep.update(status="fail", failed_stage="leakage",
                   error=f"future leakage along causal axis (max |Δ|={rep['leakage']['max_abs_diff']:.3g})")
    elif rep["lightning"].get("error"):
        rep.update(status="fail", failed_stage="lightning", error=rep["lightning"]["error"],
                   traceback=rep["lightning"].pop("traceback", None))
    rep["lightning"].pop("traceback", None)
    rep["duration_s"] = round(time.time() - t_start, 3)
    return rep


def _find_lightning_module(cfg: dict):
    """파일에서 LightningModule 하위 클래스를 찾는다. lightning.pytorch / pytorch_lightning 어느 쪽이든
    (생성 코드가 고른 패키지와 같은 Trainer를 써야 하므로) 클래스 MRO의 이름으로 판별하고 패키지를 돌려준다."""
    rel = cfg["lightning"]["file"]
    path = f"{cfg['workdir']}/{rel}"
    mod_name = rel[:-3].replace("/", ".")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    for obj in vars(module).values():
        if not isinstance(obj, type) or obj.__module__ != mod_name:
            continue
        base = next((b for b in obj.__mro__[1:] if b.__name__ == "LightningModule"), None)
        if base is not None:
            pkg = importlib.import_module(base.__module__.split(".core")[0])   # lightning.pytorch | pytorch_lightning
            return obj, pkg
    raise TypeError(f"no LightningModule subclass defined in {rel}")


def _build_lit(lit_cls, model_cls, init_kwargs: dict):
    """계약: 첫 인자로 모델 인스턴스. 계약 이전 형태(하이퍼파라미터로 스스로 모델 생성)도 받는다.
    시그니처로 먼저 고른다 — 아무 인자에나 모델 객체를 밀어 넣으면 생성은 되고 엉뚱한 곳에서 터지기 때문."""
    import inspect

    params = [p for p in inspect.signature(lit_cls.__init__).parameters.values()
              if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    names = {p.name for p in params}
    first = params[0] if params else None
    takes_model = first is not None and (
        first.name in ("model", "net", "backbone", "module", "network")
        or (isinstance(first.annotation, type) and issubclass(first.annotation, nn.Module)))
    order = []
    if takes_model:
        order.append(("lit(model)", lambda: lit_cls(model_cls(**init_kwargs))))
    if init_kwargs and set(init_kwargs) <= names:
        order.append(("lit(**init_kwargs)", lambda: lit_cls(**init_kwargs)))
    if all(p.default is not p.empty for p in params):
        order.append(("lit()", lambda: lit_cls()))
    if not order:
        raise TypeError(f"cannot construct LightningModule {lit_cls.__name__}{inspect.signature(lit_cls.__init__)}: "
                        "first argument should be the model instance (other arguments with defaults)")
    errors = []
    for how, make in order:
        try:
            torch.manual_seed(0)
            return make(), how
        except TypeError as e:
            errors.append(f"{how}: {e}")
    raise TypeError("cannot construct LightningModule — " + "; ".join(errors))


def _targets(out_spec: List[int], n: int, g: torch.Generator) -> Dict[str, torch.Tensor]:
    """명세에 target 정의가 없으므로 두 관례를 준비: 회귀(출력과 같은 shape, float) / 분류(마지막 축 = 클래스)."""
    shape = (n, *out_spec[1:])
    cands = {"regression": torch.randn(shape, generator=g)}
    if len(out_spec) >= 2 and out_spec[-1] > 1:
        cands["classification"] = torch.randint(0, out_spec[-1], shape[:-1], generator=g)
    return cands


def check_lightning(cfg: dict, model_cls) -> dict:
    """Trainer(fast_dev_run=1)로 실제 학습 경로를 1 step 돌린다. 실패 사유는 batch 형식별로 모아 보고."""
    res: Dict[str, Any] = {"checked": True, "file": cfg["lightning"]["file"], "lit_class": None, "constructed_by": None,
                           "batch_format": None, "loss": None, "params_updated": None, "validation_step": None,
                           "error": None}
    lit_cls, pl = _find_lightning_module(cfg)
    res["lit_class"] = lit_cls.__name__
    from torch.utils.data import DataLoader, TensorDataset

    b = cfg.get("batch_size") or 1
    g = torch.Generator().manual_seed(0)
    inputs = _make_inputs([{**s, "shape": [2 * b, *s["shape"][1:]]} for s in cfg["inputs"]], "cpu")
    out_spec = (cfg.get("output_expected") or [[b]])[0]
    has_val = lit_cls.validation_step is not pl.LightningModule.validation_step     # 오버라이드했을 때만 val 루프
    accelerator = {"cuda": "gpu", "mps": "mps"}.get(cfg.get("device", "cpu"), "cpu")

    failures = []
    for fmt, target in _targets(out_spec, 2 * b, g).items():
        try:
            lit, how = _build_lit(lit_cls, model_cls, cfg.get("init_kwargs", {}))
            before = {n: p.detach().clone() for n, p in lit.named_parameters() if p.requires_grad}
            losses: List[float] = []

            class _Grab(pl.Callback):
                def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                    if isinstance(loss, torch.Tensor):
                        losses.append(float(loss.detach().float().mean()) if loss.numel() == 1 else float("nan"))

            data = DataLoader(TensorDataset(*inputs, target), batch_size=b)
            trainer = pl.Trainer(fast_dev_run=1, accelerator=accelerator, devices=1, logger=False,
                                 enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                                 default_root_dir=cfg["workdir"], callbacks=[_Grab()])
            trainer.fit(lit, train_dataloaders=data, val_dataloaders=data if has_val else None)
            if not losses:
                raise RuntimeError("training_step returned no loss (None / missing 'loss' key) — nothing was optimized")
            if not all(map(lambda v: v == v and abs(v) != float("inf"), losses)):
                raise FloatingPointError(f"non-finite or non-scalar training loss: {losses}")
            updated = any(not torch.equal(before[n], p.detach().cpu()) for n, p in lit.named_parameters() if n in before)
            res.update(constructed_by=how, batch_format=fmt, loss=losses[0], params_updated=updated,
                       validation_step="ran" if has_val else "absent")
            if not updated:
                res["error"] = "optimizer step did not change any parameter (lr=0? optimizer built on wrong params?)"
            return res
        except Exception as e:  # noqa: BLE001 — 형식별 실패를 모아 둘 다 실패하면 보고
            failures.append(f"[{fmt} batch] {type(e).__name__}: {str(e)[:400]}")
            res["traceback"] = traceback.format_exc(limit=8)
    res["error"] = " | ".join(failures)
    return res


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    result = run(config)
    with open(config["result_path"], "w") as f:
        json.dump(result, f)
