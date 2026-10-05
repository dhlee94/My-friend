from my_friend.agents.schemas import ShapeRecord, ValidationReport
from ui.render import chain_conflicts, shape_table, spec_to_mermaid, validation_summary_md

from tests.graph.fixtures import tiny_spec


def test_mermaid_structure_and_chain_kinds():
    code = spec_to_mermaid(tiny_spec())
    assert code.startswith("flowchart TD")
    assert 'in_x[/"x<br>B×T×F · float32"/]:::io' in code
    # proj out (B,T,D) → conv in (B,D,T+2): rank 같고 원소 다름 → conflict 가 아니라 표시돼야 할 것은?
    # T+2 ≠ T 이므로 conflict (명세상 pad를 별도 모듈로 쓰지 않았음을 드러냄)
    assert "⚠ B×T×D ≠ B×D×T+2" in code
    assert "linkStyle 1 stroke:#dc2626" in code
    # conv out (B,D,T) → head in (B,T,D): 같은 차원 재배열 → 점선 permute
    assert 'm_conv -.->|"B×D×T → B×T×D (permute)"| m_head' in code
    assert 'm_head -->|"B×T×1"| out_y' in code


def test_chain_conflicts_lists_static_mismatch():
    (c,) = chain_conflicts(tiny_spec())
    assert "`proj` out B×T×D" in c and "`conv` in B×D×T+2" in c


def test_mermaid_colors_nodes_from_validation_and_escapes_quotes():
    spec = tiny_spec()
    spec.modules[0].role = 'say "hi"'
    rep = ValidationReport(status="fail", shape_log=[
        ShapeRecord(module="proj", output_shape="2x16x8", expected="2x16x8", ok=True),
        ShapeRecord(module="conv", output_shape="2x8x16", expected="2x16x8", ok=False),
    ])
    code = spec_to_mermaid(spec, rep)
    assert ":::ok" in code.split("m_proj[")[1].split("\n")[0]
    assert ":::bad" in code.split("m_conv[")[1].split("\n")[0]
    assert "#quot;hi#quot;" in code and 'say "hi"' not in code


def test_subgraph_grouping():
    spec = tiny_spec()
    for m, p in zip(spec.modules, ["enc.proj", "enc.conv", "head"]):
        m.path = p
    code = spec_to_mermaid(spec)
    assert 'subgraph g_enc["enc"]' in code and "m_enc_proj" in code


def test_shape_table_marks_and_repeated_calls():
    rep = ValidationReport(status="pass", shape_log=[
        ShapeRecord(module="blocks.0", output_shape="2x16x8"),
        ShapeRecord(module="shared", output_shape="2x16x8", expected="2x16x8", ok=True),
        ShapeRecord(module="shared", output_shape="2x16x8|x", expected="2x16x8", ok=False),
    ])
    t = shape_table(rep)
    lines = t.splitlines()
    assert lines[0].startswith("| # | module | call")
    assert "| `shared` | 2 |" in lines[4] and "❌" in lines[4] and "\\|x" in lines[4]
    assert shape_table(ValidationReport(status="fail")).startswith("_shape")


def test_validation_summary():
    rep = ValidationReport(status="fail", failed_stage="leakage", error="future leakage", param_count=1234,
                           shape_mismatches=["conv: expected 1, got 2"])
    rep.leakage.checked, rep.leakage.leaked = True, True
    md = validation_summary_md(rep)
    assert "FAIL" in md and "`leakage`" in md and "1,234" in md and "🚨 LEAK" in md and "- conv:" in md
