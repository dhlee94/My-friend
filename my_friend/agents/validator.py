from __future__ import annotations

from my_friend.graph.state import GraphState

from .base import AgentDeps, assistant_turn
from .schemas import MetaDetail, ValidationReport


def meta_where(d: MetaDetail) -> str:
    """'nonzero @ blocks.0.attn (model.py:42  idx = ...) [forward]' — 있는 정보만."""
    parts = [d.op or d.kind]
    if d.module:
        parts.append(f"@ {d.module}")
    if d.location:
        parts.append(f"({d.location})")
    return " ".join(parts) + f" [{d.stage}]"


def summarize(v: ValidationReport) -> str:
    head = {"pass": "✅ PASS", "fail": f"❌ FAIL @ {v.failed_stage}", "skipped": "⏭ SKIPPED"}[v.status]
    lines = [f"{head} ({v.duration_s:.1f}s)"]
    if v.error:
        lines.append(f"error: {v.error}")
    if v.status != "skipped":
        lines.append(
            f"params={v.param_count:,} ({v.param_mb:.1f}MB) · act={v.activation_mb:.1f}MB "
            f"→ train-batch 추정 {v.est_train_activation_mb:.0f}MB + Adam {v.est_adam_state_mb:.0f}MB · meta={v.meta_pass}"
        )
    if v.meta_detail:
        lines.append(f"meta pass {v.meta_detail.kind}: {meta_where(v.meta_detail)}")
    if v.leakage.checked:
        lines.append(f"leakage probe: {'LEAK' if v.leakage.leaked else 'ok'} (max|Δ|={v.leakage.max_abs_diff:.2e})")
    lt = v.lightning
    if lt.checked:
        if lt.error:
            lines.append(f"lightning ({lt.file}): ✗ {lt.error[:300]}")
        else:
            lines.append(f"lightning {lt.lit_class}: fast_dev_run ok · loss={lt.loss:.4g} · batch={lt.batch_format} "
                         f"· val_step={lt.validation_step}")
    if v.unused_params:
        lines.append(f"unused params: {', '.join(v.unused_params[:5])}")
    lines += [f"shape ✗ {m}" for m in v.shape_mismatches[:5]]
    if v.tests_status:
        lines.append(f"generated tests: {v.tests_status}" + (f" — {v.tests_detail}" if v.tests_status != "fail" else ""))
    return "\n".join(lines)


class Validator:
    """LLM 없음 — 결정적 동적 검증. 판단은 reflector 몫."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: GraphState) -> dict:
        report = self.deps.sandbox.validate(state.get("files") or {}, state.get("spec"), state.get("entry_file"))
        return {"validation": report, "messages": [assistant_turn("validator", summarize(report))]}
