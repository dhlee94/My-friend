"""샌드박스 subprocess 안에서 실행되는 검증 하네스. torch + stdlib만 의존 (my_friend import 금지).

    python shape_harness.py <config.json>   →  config["result_path"]에 JSON 리포트 기록

단계:
  1. import / build
  2. meta pass   : torch.device("meta") 위 forward — 연산 비용 0으로 shape 체인 추적.
                   nonzero/masked_select/.item() 등 data-dependent op은 meta에서 실패 → 실측으로 폴백.
  3. real pass   : torch.randn 더미 입력으로 forward + backward (NaN/Inf, unused param, activation 메모리)
  4. shape check : planner 명세의 module output shape과 실측 비교
  5. leakage     : causal 축의 t0 이후 입력을 교란 → 출력[:t0]이 변하면 미래 정보 누수
"""
from __future__ import annotations

import importlib.util
import json
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

    # ---- 2. meta pass
    if cfg.get("meta_pass", True):
        try:
            with torch.device("meta"):
                meta_model = cls(**cfg.get("init_kwargs", {}))
            meta_in = _make_inputs(cfg["inputs"], "meta")
            with torch.no_grad():
                meta_model(*meta_in)
            rep["meta_pass"] = "ok"
        except (NotImplementedError, RuntimeError) as e:
            msg = str(e)
            data_dep = any(k in msg.lower() for k in ("meta", "data-dependent", "item()", "nonzero", "cannot be"))
            rep["meta_pass"] = "data_dependent" if data_dep else f"error: {type(e).__name__}: {msg[:300]}"
        except Exception as e:  # noqa: BLE001
            rep["meta_pass"] = f"error: {type(e).__name__}: {str(e)[:300]}"

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

    if mismatches:
        rep.update(status="fail", failed_stage="shape", error="; ".join(mismatches[:5]))
    elif rep["nan_detected"]:
        rep.update(status="fail", failed_stage="numerical", error="NaN/Inf in forward output")
    elif rep["leakage"]["leaked"]:
        rep.update(status="fail", failed_stage="leakage",
                   error=f"future leakage along causal axis (max |Δ|={rep['leakage']['max_abs_diff']:.3g})")
    rep["duration_s"] = round(time.time() - t_start, 3)
    return rep


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    result = run(config)
    with open(config["result_path"], "w") as f:
        json.dump(result, f)
