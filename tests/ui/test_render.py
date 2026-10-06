from my_friend.agents.schemas import ModuleSpec, ShapeRecord, ValidationReport
from ui.render import chain_conflicts, shape_table, spec_to_mermaid, validation_summary_md, workflow_mermaid

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


def test_workflow_mermaid_sets_text_color_for_dark_theme():
    raw = ("graph TD;\n\t__start__([<p>__start__</p>]):::first\n\trouter(router)\n"
           "\tclassDef default fill:#f2f0ff,line-height:1.2\n\tclassDef first fill-opacity:0\n"
           "\tclassDef last fill:#bfb6fc\n\tclassDef mine fill:#fff,color:#000\n")
    out = workflow_mermaid(raw)
    assert "<p>" not in out and "__start__([__start__])" in out
    assert "classDef default fill:#f2f0ff,line-height:1.2,color:#1e1b4b" in out
    assert "classDef last fill:#bfb6fc,color:#1e1b4b" in out
    assert "classDef first fill:#f2f0ff,color:#1e1b4b" in out   # 투명 __start__도 밝은 배경 + 어두운 글자
    assert "classDef mine fill:#fff,color:#000" in out          # 이미 글자색이 있으면 그대로


def _layered(n_layers, mlp_out=("B", "T", "D"), odd=None):
    mods = []
    for i in range(n_layers):
        out = ["B", "T", "D*2"] if odd == i else list(mlp_out)
        mods += [ModuleSpec(path=f"encoder.layers.{i}.attn", role=f"layer {i} self-attention " + "x" * 80,
                            input_shape=["B", "T", "D"], output_shape=["B", "T", "D"], complexity="O(B·T²·D)"),
                 ModuleSpec(path=f"encoder.layers.{i}.mlp", role="mlp", input_shape=["B", "T", "D"],
                            output_shape=out, complexity="O(B·T·D²)")]
    return tiny_spec(name="Big", modules=mods)


def test_repeated_layers_collapse_into_one_block():
    code = spec_to_mermaid(_layered(16))
    assert code.count(":::mod") == 2                                  # 32개 모듈 → 대표 2개
    assert 'encoder.layers.[0‥15].attn ×16' in code and 'encoder.layers.[0‥15].mlp ×16' in code
    assert "×16 반복" in code                                         # mlp → attn 되돌아가는 반복 엣지
    assert "x" * 70 not in code and "…" in code                       # 긴 role은 잘림


def test_collapse_needs_identical_structure_and_min_repeats():
    code = spec_to_mermaid(_layered(6, odd=5))                        # layer 5만 mlp 출력이 다름
    assert "encoder.layers.[0‥4].attn ×5" in code
    assert "encoder.layers.5.attn" in code and "encoder.layers.5.mlp" in code
    code = spec_to_mermaid(_layered(2))                               # 2회 반복은 그대로 펼친다
    assert "‥" not in code and code.count(":::mod") == 4


def test_collapsed_node_is_red_if_any_repeat_mismatches():
    spec = _layered(4)
    log = [ShapeRecord(module=f"encoder.layers.{i}.{part}", output_shape="2x16x8", expected="2x16x8",
                       ok=not (i == 2 and part == "attn")) for i in range(4) for part in ("attn", "mlp")]
    code = spec_to_mermaid(spec, ValidationReport(status="fail", shape_log=log))
    attn = next(l for l in code.splitlines() if "[0‥3].attn" in l)
    mlp = next(l for l in code.splitlines() if "[0‥3].mlp" in l)
    assert attn.endswith(":::bad") and mlp.endswith(":::ok")
    assert "✗ 불일치: 2" in attn and "불일치" not in mlp


def test_repeat_edge_flags_chain_conflict_between_repeats():
    code = spec_to_mermaid(_layered(3, mlp_out=("B", "T", "D*2")))   # mlp 출력이 다음 반복 attn 입력과 불일치
    loop = next(l for l in code.splitlines() if "반복" in l)
    assert "⚠ B×T×D*2 ≠ B×T×D" in loop
    assert any(l.strip().startswith("linkStyle") for l in code.splitlines())
