"""UI 렌더링 순수 함수 (Chainlit 비의존 → 단위 테스트 가능).

- spec_to_mermaid : ModelSpec → Mermaid flowchart. 모듈 간 shape 체인을 '정적으로' 검사해
                    전치/reshape은 점선, 차원 충돌은 붉은 ⚠ 엣지로 표시 (Shape Tracker).
- shape_table     : ValidationReport.shape_log → 마크다운 표 (실측 vs 명세)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

from my_friend.agents.schemas import ModelSpec, ValidationReport


def _sid(prefix: str, name: str) -> str:
    return f"{prefix}_{re.sub(r'[^0-9A-Za-z_]', '_', name)}"


def _q(text: str) -> str:
    """Mermaid 따옴표 라벨 이스케이프."""
    return text.replace('"', "#quot;").replace("\n", "<br>")


def _shape(dims: List[str]) -> str:
    return "×".join(str(d).replace(" ", "") for d in dims)


def _chain_kind(prev_out: List[str], nxt_in: List[str]) -> str:
    """ok | permute (같은 차원의 재배열: transpose/permute) | reshape (원소 수 보존 가능성) | conflict"""
    a = [str(x).replace(" ", "") for x in prev_out]
    b = [str(x).replace(" ", "") for x in nxt_in]
    if a == b:
        return "ok"
    if Counter(a) == Counter(b):
        return "permute"
    if a and b and a[0] == b[0] and len(a) != len(b):
        return "reshape"            # 배치 축 유지 + rank 변화 → flatten/view 계열로 간주
    return "conflict"


def spec_to_mermaid(spec: ModelSpec, report: Optional[ValidationReport] = None) -> str:
    """모듈은 spec.modules 나열 순서(=forward 순서)로 연결한다.

    report가 있으면 실측 결과로 노드 색을 칠한다 (ok=green, mismatch=red).
    """
    observed: Dict[str, Optional[bool]] = {}
    if report is not None:
        for r in report.shape_log:
            observed.setdefault(r.module, r.ok)

    lines = ["flowchart TD"]
    lines += [
        "  classDef io fill:#eef2ff,stroke:#6366f1,color:#1e1b4b",
        "  classDef ok fill:#dcfce7,stroke:#16a34a,color:#052e16",
        "  classDef bad fill:#fee2e2,stroke:#dc2626,color:#450a0a",
        "  classDef mod fill:#f8fafc,stroke:#64748b,color:#0f172a",
    ]

    for t in spec.inputs:
        lines.append(f'  {_sid("in", t.name)}[/"{_q(t.name)}<br>{_q(_shape(t.shape))} · {t.dtype}"/]:::io')

    # 최상위 경로 접두사가 같은 모듈이 2개 이상이면 subgraph로 묶는다 (encoder.*, decoder.*)
    groups: Dict[str, List[int]] = {}
    for i, m in enumerate(spec.modules):
        groups.setdefault(m.path.split(".")[0], []).append(i)
    grouped = {i for idxs in groups.values() if len(idxs) > 1 for i in idxs}

    def node_line(i: int) -> str:
        m = spec.modules[i]
        ok = observed.get(m.path)
        cls = "mod" if ok is None else ("ok" if ok else "bad")
        # securityLevel=strict: <br> 외 HTML 태그는 이스케이프되므로 순수 텍스트만 사용
        return f'  {_sid("m", m.path)}["{_q(m.path)}<br>{_q(m.role)}<br>{_q(m.complexity)}"]:::{cls}'

    emitted = set()
    for i in range(len(spec.modules)):
        if i in emitted:
            continue
        head = spec.modules[i].path.split(".")[0]
        if i in grouped:
            lines.append(f'  subgraph {_sid("g", head)}["{_q(head)}"]')
            for j in groups[head]:
                lines.append("  " + node_line(j))
                emitted.add(j)
            lines.append("  end")
        else:
            lines.append(node_line(i))
            emitted.add(i)

    for t in spec.outputs:
        lines.append(f'  {_sid("out", t.name)}[/"{_q(t.name)}<br>{_q(_shape(t.shape))}"/]:::io')

    # ---- edges
    link_idx = 0
    styles: List[str] = []
    mods = spec.modules
    if mods:
        first = mods[0]
        for t in spec.inputs:
            lines.append(f'  {_sid("in", t.name)} -->|"{_q(_shape(t.shape))}"| {_sid("m", first.path)}')
            link_idx += 1
        for prev, nxt in zip(mods, mods[1:]):
            kind = _chain_kind(prev.output_shape, nxt.input_shape)
            a, b = _shape(prev.output_shape), _shape(nxt.input_shape)
            src, dst = _sid("m", prev.path), _sid("m", nxt.path)
            if kind == "ok":
                lines.append(f'  {src} -->|"{_q(a)}"| {dst}')
            elif kind in ("permute", "reshape"):
                lines.append(f'  {src} -.->|"{_q(a)} → {_q(b)} ({kind})"| {dst}')
            else:
                lines.append(f'  {src} -->|"⚠ {_q(a)} ≠ {_q(b)}"| {dst}')
                styles.append(f"  linkStyle {link_idx} stroke:#dc2626,stroke-width:2px,color:#dc2626")
            link_idx += 1
        last = mods[-1]
        for t in spec.outputs:
            lines.append(f'  {_sid("m", last.path)} -->|"{_q(_shape(last.output_shape))}"| {_sid("out", t.name)}')
            link_idx += 1
    else:
        for i_t in spec.inputs:
            for o_t in spec.outputs:
                lines.append(f'  {_sid("in", i_t.name)} --> {_sid("out", o_t.name)}')
    return "\n".join(lines + styles)


def chain_conflicts(spec: ModelSpec) -> List[str]:
    """승인 전에 사용자에게 보여줄 정적 shape 충돌 목록."""
    out = []
    for prev, nxt in zip(spec.modules, spec.modules[1:]):
        if _chain_kind(prev.output_shape, nxt.input_shape) == "conflict":
            out.append(f"`{prev.path}` out {_shape(prev.output_shape)} ≠ `{nxt.path}` in {_shape(nxt.input_shape)}")
    return out


def _cell(s: str) -> str:
    return s.replace("|", "\\|")


def shape_table(report: ValidationReport, max_rows: int = 120) -> str:
    """레이어별 실측 shape 표. 같은 모듈의 반복 호출(루프/공유 가중치)은 call 번호로 구분."""
    if not report.shape_log:
        return "_shape 로그 없음 (forward 이전 단계에서 실패)_"
    calls: Counter = Counter()
    rows = ["| # | module | call | output shape | expected | |", "|---:|---|---:|---|---|:---:|"]
    for i, r in enumerate(report.shape_log[:max_rows]):
        calls[r.module] += 1
        mark = "" if r.ok is None else ("✅" if r.ok else "❌")
        exp = r.expected or ""
        rows.append(f"| {i} | `{_cell(r.module)}` | {calls[r.module]} | `{_cell(r.output_shape)}` | {_cell(exp)} | {mark} |")
    if len(report.shape_log) > max_rows:
        rows.append(f"| … | _{len(report.shape_log) - max_rows} rows more_ | | | | |")
    return "\n".join(rows)


def validation_summary_md(v: ValidationReport) -> str:
    head = {"pass": "✅ **PASS**", "fail": f"❌ **FAIL** @ `{v.failed_stage}`", "skipped": "⏭ **SKIPPED**"}[v.status]
    lines = [f"{head} · {v.duration_s:.1f}s"]
    if v.error:
        lines.append(f"> {v.error}")
    if v.status != "skipped":
        lines.append(
            "| params | param mem | activation (val) | activation @train batch | Adam state | meta pass |\n"
            "|---:|---:|---:|---:|---:|---|\n"
            f"| {v.param_count:,} | {v.param_mb:.1f} MB | {v.activation_mb:.2f} MB | "
            f"{v.est_train_activation_mb:.0f} MB | {v.est_adam_state_mb:.0f} MB | {v.meta_pass} |"
        )
    if v.leakage.checked:
        lines.append(f"**Leakage probe:** {'🚨 LEAK' if v.leakage.leaked else 'ok'} (max\\|Δ\\|={v.leakage.max_abs_diff:.2e}) — {v.leakage.detail}")
    if v.unused_params:
        lines.append("**Unused params:** " + ", ".join(f"`{p}`" for p in v.unused_params[:8]))
    if v.shape_mismatches:
        lines.append("**Shape mismatches:**\n" + "\n".join(f"- {m}" for m in v.shape_mismatches[:10]))
    return "\n\n".join(lines)
