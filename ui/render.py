"""UI 렌더링 순수 함수 (Chainlit 비의존 → 단위 테스트 가능).

- spec_to_mermaid : ModelSpec → Mermaid flowchart. 모듈 간 shape 체인을 '정적으로' 검사해
                    전치/reshape은 점선, 차원 충돌은 붉은 ⚠ 엣지로 표시 (Shape Tracker).
                    반복 레이어는 접어서 큰 명세도 한 화면 안팎으로 (#4).
- shape_table     : ValidationReport.shape_log → 마크다운 표 (실측 vs 명세)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from my_friend.agents.schemas import ModelSpec, ValidationReport
from my_friend.tools.sandbox import resolve_dim


def _sid(prefix: str, name: str) -> str:
    return f"{prefix}_{re.sub(r'[^0-9A-Za-z_]', '_', name)}"


def _q(text: str) -> str:
    """Mermaid 따옴표 라벨 이스케이프."""
    return text.replace('"', "#quot;").replace("\n", "<br>")


def _shape(dims: List[str]) -> str:
    return "×".join(str(d).replace(" ", "") for d in dims)


def _numel(dims: List[str], values: Optional[Dict[str, int]]) -> Optional[int]:
    """명세 검증값으로 원소 수 계산. 값이 없거나 모르는 기호면 None."""
    if not values:
        return None
    n = 1
    try:
        for d in dims:
            n *= resolve_dim(str(d), values)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return None
    return n


def _chain_kind(prev_out: List[str], nxt_in: List[str], values: Optional[Dict[str, int]] = None) -> str:
    """ok | permute (같은 차원의 재배열: transpose/permute) | reshape (원소 수 보존) | conflict

    rank가 바뀌는 연결은 명세 검증값(values)으로 원소 수를 비교한다: 같으면 flatten/view(reshape),
    다르면 conflict. 값을 못 구하면 예전처럼 배치 축 유지 + rank 변화를 reshape로 간주한다.
    rank가 같은데 차원이 다르면 원소 수가 우연히 같아도(작은 검증값) conflict로 둔다.
    """
    a = [str(x).replace(" ", "") for x in prev_out]
    b = [str(x).replace(" ", "") for x in nxt_in]
    if a == b:
        return "ok"
    if Counter(a) == Counter(b):
        return "permute"
    if a and b and a[0] == b[0] and len(a) != len(b):
        na, nb = _numel(a, values), _numel(b, values)
        if na is not None and nb is not None and na != nb:
            return "conflict"
        return "reshape"
    return "conflict"


def _numel_note(a: List[str], b: List[str], values: Optional[Dict[str, int]]) -> str:
    """rank가 다른 conflict에 원소 수를 덧붙여 왜 reshape가 아닌지 보인다."""
    if len(a) == len(b):
        return ""
    na, nb = _numel(a, values), _numel(b, values)
    return f" (원소 수 {na} ≠ {nb})" if na is not None and nb is not None and na != nb else ""


def workflow_mermaid(code: str) -> str:
    """LangGraph draw_mermaid() 출력을 UI용으로 정리.
    - <p> 라벨 제거: securityLevel=strict에선 태그가 글자로 보인다.
    - classDef에 글자색 지정: LangGraph는 밝은 fill만 주므로 dark 테마에서 글자(밝은 기본색)가 배경에 묻힌다.
    - 투명 노드(__start__)에도 밝은 fill: 상속된 어두운 글자색이 dark 배경에 묻히지 않게.
    """
    code = re.sub(r"</?p>", "", code)
    # default의 글자색은 모든 노드에 상속된다 → 투명 배경(__start__)은 dark 테마에서 어두운 글자가 묻히므로 밝은 fill을 준다
    code = re.sub(r"^(\s*classDef \S+ )fill-opacity:0\s*$", r"\1fill:#f2f0ff", code, flags=re.M)
    return re.sub(r"^(\s*classDef \S+ (?!.*color:).*fill:[^\n]*)$", r"\1,color:#1e1b4b", code, flags=re.M)


_INDEXED = re.compile(r"^(?P<prefix>.+?)\.(?P<idx>\d+)(?:\.(?P<rest>.+))?$")
COLLAPSE_MIN_REPEATS = 3          # 이만큼 연속으로 같은 구조인 레이어는 하나로 접는다
ROLE_MAX_CHARS = 60               # 노드 라벨의 role 길이 (전문은 명세 표에 있다)


@dataclass
class _Node:
    """다이어그램 노드 1개 = 모듈 1개, 또는 같은 구조로 반복되는 레이어들의 같은 위치 모듈 묶음."""

    node_id: str
    path: str                     # 표시용 경로 (접힌 경우 "encoder.layers.[0‥15].attn")
    head: str                     # subgraph 묶음 기준 (최상위 경로)
    role: str
    complexity: str
    input_shape: List[str]
    output_shape: List[str]
    members: List[str]            # 실측 색칠에 쓰는 원래 모듈 경로들
    repeat: int = 1


def _short(text: str, n: int = ROLE_MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _layout_nodes(spec: ModelSpec) -> List[Tuple[List[_Node], int]]:
    """modules를 (노드 묶음, 반복 횟수) 시퀀스로. 반복 1 = 펼친 그대로, n>1 = 같은 구조 n회를 대표 1개로 접음."""
    mods = spec.modules
    out: List[Tuple[List[_Node], int]] = []

    def plain(i: int) -> _Node:
        m = mods[i]
        return _Node(_sid("m", m.path), m.path, m.path.split(".")[0], m.role, m.complexity,
                     m.input_shape, m.output_shape, [m.path])

    i = 0
    while i < len(mods):
        hit = _INDEXED.match(mods[i].path)
        if not hit:
            out.append(([plain(i)], 1))
            i += 1
            continue
        # 같은 prefix로 이어지는 구간을 index별 유닛으로 자른다: encoder.layers.0.{attn,mlp}, encoder.layers.1.{…}
        prefix = hit["prefix"]
        units: List[Tuple[int, List[int]]] = []
        j = i
        while j < len(mods):
            h = _INDEXED.match(mods[j].path)
            if not h or h["prefix"] != prefix:
                break
            idx = int(h["idx"])
            if units and units[-1][0] == idx:
                units[-1][1].append(j)
            elif units and idx != units[-1][0] + 1:
                break                                  # index가 끊기면 다른 구간
            else:
                units.append((idx, [j]))
            j += 1

        def sig(unit: List[int]) -> tuple:
            return tuple((_INDEXED.match(mods[k].path)["rest"], tuple(mods[k].input_shape),
                          tuple(mods[k].output_shape), mods[k].complexity) for k in unit)

        u = 0
        while u < len(units):
            v = u + 1
            while v < len(units) and sig(units[v][1]) == sig(units[u][1]):
                v += 1
            run = units[u:v]
            if len(run) >= COLLAPSE_MIN_REPEATS:
                first_idx, last_idx = run[0][0], run[-1][0]
                nodes = []
                for pos, k in enumerate(run[0][1]):
                    m, rest = mods[k], _INDEXED.match(mods[k].path)["rest"]
                    shown = f"{prefix}.[{first_idx}‥{last_idx}]" + (f".{rest}" if rest else "")
                    nodes.append(_Node(_sid("m", f"{prefix}.x.{rest or ''}"), shown, prefix.split(".")[0], m.role,
                                       m.complexity, m.input_shape, m.output_shape,
                                       [mods[unit[pos]].path for _, unit in run], repeat=len(run)))
                out.append((nodes, len(run)))
            else:
                for _, unit in run:
                    out.append(([plain(k) for k in unit], 1))
            u = v
        i = j
    return out


def spec_to_mermaid(spec: ModelSpec, report: Optional[ValidationReport] = None) -> str:
    """모듈은 spec.modules 나열 순서(=forward 순서)로 연결한다.

    report가 있으면 실측 결과로 노드 색을 칠한다 (ok=green, mismatch=red).
    같은 구조로 COLLAPSE_MIN_REPEATS회 이상 반복되는 인덱스 레이어(encoder.layers.N.*)는 대표 하나로 접고
    "×n" 반복 엣지를 단다. 접힌 노드 색은 모든 반복의 실측을 합친다 (하나라도 불일치면 red).
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

    layout = _layout_nodes(spec)
    nodes = [n for group, _ in layout for n in group]

    # 최상위 경로 접두사가 같은 노드가 2개 이상이면 subgraph로 묶는다 (encoder.*, decoder.*)
    groups: Dict[str, List[int]] = {}
    for i, n in enumerate(nodes):
        groups.setdefault(n.head, []).append(i)
    grouped = {i for idxs in groups.values() if len(idxs) > 1 for i in idxs}

    def node_line(n: _Node) -> str:
        seen = [observed[p] for p in n.members if observed.get(p) is not None]
        cls = "mod" if not seen else ("ok" if all(seen) else "bad")
        title = n.path + (f" ×{n.repeat}" if n.repeat > 1 else "")
        label = [title, _short(n.role), n.complexity]
        if n.repeat > 1 and cls == "bad":       # 접힌 노드: 어느 반복이 틀렸는지 드러낸다
            bad = [_INDEXED.match(p)["idx"] for p in n.members if observed.get(p) is False]
            label.append("✗ 불일치: " + ", ".join(bad[:8]) + (" …" if len(bad) > 8 else ""))
        # securityLevel=strict: <br> 외 HTML 태그는 이스케이프되므로 순수 텍스트만 사용
        return f'  {n.node_id}["{"<br>".join(_q(x) for x in label)}"]:::{cls}'

    emitted = set()
    for i, n in enumerate(nodes):
        if i in emitted:
            continue
        if i in grouped:
            lines.append(f'  subgraph {_sid("g", n.head)}["{_q(n.head)}"]')
            for j in groups[n.head]:
                lines.append("  " + node_line(nodes[j]))
                emitted.add(j)
            lines.append("  end")
        else:
            lines.append(node_line(n))
            emitted.add(i)

    for t in spec.outputs:
        lines.append(f'  {_sid("out", t.name)}[/"{_q(t.name)}<br>{_q(_shape(t.shape))}"/]:::io')

    # ---- edges
    values = spec.dim_values()
    link_idx = 0
    styles: List[str] = []

    def edge(src: str, out_dims: List[str], dst: str, in_dims: List[str], extra: str = "") -> None:
        nonlocal link_idx
        kind = _chain_kind(out_dims, in_dims, values)
        a, b = _shape(out_dims), _shape(in_dims)
        if kind == "ok":
            lines.append(f'  {src} -->|"{_q(a)}{extra}"| {dst}')
        elif kind in ("permute", "reshape"):
            lines.append(f'  {src} -.->|"{_q(a)} → {_q(b)} ({kind}){extra}"| {dst}')
        else:
            lines.append(f'  {src} -->|"⚠ {_q(a)} ≠ {_q(b)}{_numel_note(out_dims, in_dims, values)}{extra}"| {dst}')
            # 라벨 글자 #b91c1c: 라이트 라벨 배경(#e8e8e8) 위 5.3:1. 다크에서는 Mermaid.jsx가 밝은 빨강으로 바꾼다
            styles.append(f"  linkStyle {link_idx} stroke:#dc2626,stroke-width:2px,color:#b91c1c")
        link_idx += 1

    if nodes:
        first, last = nodes[0], nodes[-1]
        for t in spec.inputs:
            if len(spec.inputs) == 1:      # 입력이 하나면 첫 모듈 입력과 체인 검사 (여럿이면 어느 것이 들어가는지 명세에 없음)
                edge(_sid("in", t.name), t.shape, first.node_id, first.input_shape)
            else:
                lines.append(f'  {_sid("in", t.name)} -->|"{_q(_shape(t.shape))}"| {first.node_id}')
                link_idx += 1
        for prev, nxt in zip(nodes, nodes[1:]):
            edge(prev.node_id, prev.output_shape, nxt.node_id, nxt.input_shape)
        for group, repeat in layout:
            if repeat > 1:                 # 반복 블록: 마지막 → 첫 노드로 되돌아가는 엣지 (반복 사이 shape 체인도 검사)
                edge(group[-1].node_id, group[-1].output_shape, group[0].node_id, group[0].input_shape,
                     extra=f" · ×{repeat} 반복")
        for t in spec.outputs:
            if len(spec.outputs) == 1:
                edge(last.node_id, last.output_shape, _sid("out", t.name), t.shape)
            else:
                lines.append(f'  {last.node_id} -->|"{_q(_shape(last.output_shape))}"| {_sid("out", t.name)}')
                link_idx += 1
    else:
        for i_t in spec.inputs:
            for o_t in spec.outputs:
                lines.append(f'  {_sid("in", i_t.name)} --> {_sid("out", o_t.name)}')
    return "\n".join(lines + styles)


def chain_conflicts(spec: ModelSpec) -> List[str]:
    """승인 전에 사용자에게 보여줄 정적 shape 충돌 목록.
    입력/출력 텐서가 하나뿐이면 입력 → 첫 모듈, 마지막 모듈 → 출력 연결도 검사한다."""
    out = []
    mods, values = spec.modules, spec.dim_values()

    def check(a: List[str], b: List[str], left: str, right: str) -> None:
        if _chain_kind(a, b, values) == "conflict":
            out.append(f"{left} {_shape(a)} ≠ {right} {_shape(b)}{_numel_note(a, b, values)}")

    if mods and len(spec.inputs) == 1:
        t, first = spec.inputs[0], mods[0]
        check(t.shape, first.input_shape, f"입력 `{t.name}`", f"`{first.path}` in")
    for prev, nxt in zip(mods, mods[1:]):
        check(prev.output_shape, nxt.input_shape, f"`{prev.path}` out", f"`{nxt.path}` in")
    if mods and len(spec.outputs) == 1:
        last, t = mods[-1], spec.outputs[0]
        check(last.output_shape, t.shape, f"`{last.path}` out", f"출력 `{t.name}`")
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
    lt = v.lightning
    if lt.checked:
        lines.append(f"**Lightning** `{lt.lit_class or lt.file}`: " + (
            f"❌ {lt.error}" if lt.error else
            f"✅ fast_dev_run · loss={lt.loss:.4g} · batch `{lt.batch_format}` · validation_step {lt.validation_step}"))
    if v.unused_params:
        lines.append("**Unused params:** " + ", ".join(f"`{p}`" for p in v.unused_params[:8]))
    if v.shape_mismatches:
        lines.append("**Shape mismatches:**\n" + "\n".join(f"- {m}" for m in v.shape_mismatches[:10]))
    if v.tests_status == "fail":
        lines.append("**Generated tests:** ❌ fail\n```\n" + v.tests_detail[-1500:] + "\n```")
    elif v.tests_status:
        lines.append(f"**Generated tests:** {'✅' if v.tests_status == 'pass' else '⏭'} {v.tests_status} — {v.tests_detail}")
    return "\n\n".join(lines)
