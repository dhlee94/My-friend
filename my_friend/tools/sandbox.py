"""생성 코드 동적 검증 샌드박스.

현재 격리 수준: 별도 프로세스 + 임시 디렉토리 + timeout + 비밀 env 제거.
LLM 생성 코드를 실행하므로 이것은 '사고 방지' 수준이지 보안 경계가 아니다 —
서버 배포 시에는 `docker run --network none --memory ... --pids-limit ...`로 감싼 DockerSandbox로 교체.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import py_compile
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from my_friend.agents.schemas import ModelSpec, ValidationReport

_HARNESS = Path(__file__).with_name("shape_harness.py")
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv, ast.Div: operator.floordiv, ast.Pow: operator.pow}


def resolve_dim(expr: str, dims: Dict[str, int]) -> int:
    """'D*4', 'T//2', 'H*W', '3' → int. 산술 외 구문은 거부 (eval 미사용)."""

    def ev(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in dims:
                raise ValueError(f"unknown dim symbol '{n.id}' in '{expr}'")
            return dims[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(f"unsupported dim expression '{expr}'")

    return ev(ast.parse(str(expr).strip(), mode="eval"))


def build_harness_config(spec: ModelSpec, entry_file: str, device: str = "cpu") -> dict:
    dims = spec.dim_values()
    inputs = [{"name": t.name, "shape": [resolve_dim(s, dims) for s in t.shape], "dtype": t.dtype}
              for t in spec.inputs]
    causal = None
    if spec.causal_axis:
        out_shape = spec.outputs[0].shape if spec.outputs else []
        if spec.causal_axis in out_shape:
            causal = {
                "input_axes": [t.shape.index(spec.causal_axis) if spec.causal_axis in t.shape else None
                               for t in spec.inputs],
                "output_axis": out_shape.index(spec.causal_axis),
            }
    return {
        "entry_file": entry_file,
        "entry_class": spec.entry_class,
        "init_kwargs": spec.init_kwargs(),
        "inputs": inputs,
        "expected": [{"module": m.path, "shape": [resolve_dim(s, dims) for s in m.output_shape]}
                     for m in spec.modules],
        "output_expected": [[resolve_dim(s, dims) for s in t.shape] for t in spec.outputs[:1]],
        "causal": causal,
        "batch_size": dims.get("B") or (inputs[0]["shape"][0] if inputs and inputs[0]["shape"] else 1),
        "train_batch_size": spec.train_batch_size,
        "device": device,
        "meta_pass": True,
    }


class Sandbox(Protocol):
    def validate(self, files: Dict[str, str], spec: Optional[ModelSpec], entry_file: Optional[str]) -> ValidationReport: ...


class SubprocessSandbox:
    def __init__(self, python: str = sys.executable, timeout_s: float = 120.0, device: str = "cpu") -> None:
        # 기본 cpu: 결정적이고 MPS 미지원 op에 의한 거짓 실패가 없다. 성능 검증은 별도 단계.
        self.python, self.timeout_s, self.device = python, timeout_s, device

    def validate(self, files: Dict[str, str], spec: Optional[ModelSpec], entry_file: Optional[str]) -> ValidationReport:
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="myfriend_sbx_") as tmp:
            root = Path(tmp)
            for rel, src in files.items():
                dst = (root / rel).resolve()
                if root.resolve() not in dst.parents:
                    return ValidationReport(status="fail", failed_stage="compile", error=f"path escapes sandbox: {rel}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src)

            report = self._harness(root, files, spec, entry_file, t0)
            tests = sorted(r for r in files if _is_test_file(r))
            if tests and report.status in ("pass", "skipped"):
                # 하네스(shape/누수/수치)가 통과한 코드에 한해 생성 테스트를 돌린다: 실패 원인을 섞지 않기 위해
                self._run_tests(root, tests, report)
            report.duration_s = round(time.time() - t0, 3)
            return report

    def _harness(self, root: Path, files: Dict[str, str], spec: Optional[ModelSpec],
                 entry_file: Optional[str], t0: float) -> ValidationReport:
        errors = _compile_all(root, [r for r in files if r.endswith(".py")])
        if errors:
            return ValidationReport(status="fail", failed_stage="compile", error=errors[0],
                                    traceback="\n".join(errors), duration_s=round(time.time() - t0, 3))
        if spec is None or not entry_file:
            return ValidationReport(status="skipped", error="no ModelSpec: compiled only, forward not executed",
                                    duration_s=round(time.time() - t0, 3))
        try:
            cfg = build_harness_config(spec, entry_file, self.device)
        except ValueError as e:
            return ValidationReport(status="fail", failed_stage="spec", error=str(e))
        cfg["workdir"] = str(root)
        cfg["result_path"] = str(root / "_result.json")
        (root / "_config.json").write_text(json.dumps(cfg))

        try:
            proc = subprocess.run(
                [self.python, str(_HARNESS), str(root / "_config.json")],
                cwd=root, env=_sandbox_env(), capture_output=True, text=True, timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ValidationReport(status="fail", failed_stage="timeout",
                                    error=f"validation exceeded {self.timeout_s}s (infinite loop or too-large dims?)",
                                    duration_s=round(time.time() - t0, 3))
        result_path = root / "_result.json"
        if not result_path.exists():
            # 하네스 자체가 죽음: segfault, OOM kill(-9) 등
            return ValidationReport(status="fail", failed_stage="crash",
                                    error=f"harness exited with code {proc.returncode}",
                                    stderr_tail=proc.stderr[-2000:], duration_s=round(time.time() - t0, 3))
        data = json.loads(result_path.read_text())
        data["stderr_tail"] = proc.stderr[-2000:]
        return ValidationReport.model_validate(data)

    def _run_tests(self, root: Path, tests: List[str], report: ValidationReport) -> None:
        """pytest 결과를 report에 기록. 테스트 실패 = 검증 실패(failed_stage="tests") → reflector가
        모델이 틀렸는지 테스트가 틀렸는지 판단한다."""
        cmd = [self.python, "-m", "pytest", "-q", "-x", "-rf", "-p", "no:cacheprovider", "--no-header", *tests]
        try:
            proc = subprocess.run(cmd, cwd=root, env=_sandbox_env(), capture_output=True, text=True,
                                  timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            report.status, report.failed_stage = "fail", "tests"
            report.tests_status = "fail"
            report.error = f"generated tests exceeded {self.timeout_s}s"
            return
        out = (proc.stdout + proc.stderr)[-3000:]
        if proc.returncode == 0:
            report.tests_status = "pass"
            report.tests_detail = (proc.stdout.strip().splitlines() or [""])[-1]       # "3 passed in 0.4s"
        elif "No module named pytest" in proc.stderr:
            report.tests_status, report.tests_detail = "not_run", "pytest가 샌드박스 인터프리터에 없음"
        elif proc.returncode == 5:
            report.tests_status, report.tests_detail = "not_run", "수집된 테스트 없음"
        elif proc.returncode in (1, 2):                 # 1 = 테스트 실패, 2 = 수집 오류(테스트 파일 import 실패 등)
            failed = [l for l in proc.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))]
            report.status, report.failed_stage = "fail", "tests"
            report.tests_status, report.tests_detail = "fail", out
            report.error = "; ".join(failed[:3]) or f"pytest exited with code {proc.returncode}"
        else:                                           # 3/4 = pytest 내부/사용법 오류: 생성 코드 탓이 아님
            report.tests_status, report.tests_detail = "not_run", f"pytest exited with code {proc.returncode}\n{out}"


def _is_test_file(rel: str) -> bool:
    name = Path(rel).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _sandbox_env() -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in _SECRET_ENV_MARKERS)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _compile_all(root: Path, rels: List[str]) -> List[str]:
    errors = []
    for rel in rels:
        try:
            py_compile.compile(str(root / rel), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"{rel}: {e.msg.strip()}")
    return errors
