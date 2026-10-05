from my_friend.rag.chunker import MarkdownChunker, parse_blocks
from my_friend.rag.schema import classify_section

from .conftest import PAPER


def test_parse_blocks_protects_math_table_code():
    kinds = [b.kind for b in parse_blocks(PAPER)]
    assert kinds.count("equation") == 1
    assert kinds.count("code") == 1
    assert kinds.count("table") == 1
    eq = next(b for b in parse_blocks(PAPER) if b.kind == "equation")
    assert eq.text.strip().startswith("$$") and eq.text.strip().endswith("$$")


def test_table_caption_is_attached():
    table = next(b for b in parse_blocks(PAPER) if b.kind == "table")
    assert table.text.startswith("Table 2:")
    assert "Transformer (big)" in table.text


def test_breadcrumb_and_section_type():
    chunks = MarkdownChunker().chunk(PAPER, doc_id="vaswani17")
    by_crumb = {tuple(c.breadcrumb): c for c in chunks}
    attn = by_crumb[("3 Model Architecture", "3.2 Scaled Dot-Product Attention")]
    assert attn.doc_title == "Attention Is All You Need"
    assert attn.section_type == "methodology"
    assert attn.indexed_text.startswith("[Attention Is All You Need > 3 Model Architecture > 3.2 Scaled")
    assert by_crumb[("6 Experiments",)].section_type == "experiments"
    assert by_crumb[("Abstract",)].section_type == "abstract"


def test_chunks_never_cross_sections():
    chunks = MarkdownChunker().chunk(PAPER, doc_id="d")
    for c in chunks:
        assert "Positional Encoding" not in c.content  # 헤더는 본문이 아니라 breadcrumb에만


def test_equation_never_split_under_tiny_budget():
    # 예산을 수식보다 작게 → 수식은 단독 oversized 청크가 되어야지 잘리면 안 된다
    chunker = MarkdownChunker(max_tokens=20, overlap_tokens=5)
    chunks = chunker.chunk(PAPER, doc_id="d")
    eq_chunks = [c for c in chunks if "equation" in c.block_types]
    assert len(eq_chunks) >= 1
    for c in chunks:
        assert c.content.count("$$") % 2 == 0, c.content
        assert c.content.count("```") % 2 == 0, c.content
    assert any(c.oversized for c in eq_chunks)
    table_chunks = [c for c in chunks if "table" in c.block_types]
    assert all("| Transformer (big)" in c.content for c in table_chunks)


def test_overlap_carries_context_within_section():
    text = "## Method\n\n" + "\n\n".join(f"Sentence number {i} describes the method in detail." for i in range(30))
    chunks = MarkdownChunker(max_tokens=60, overlap_tokens=15).chunk(text, doc_id="d", doc_title="T")
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:]):
        last_para = a.content.split("\n\n")[-1]
        assert b.content.startswith(last_para)
        assert b.token_count <= 60 + 5  # 블록 결합 개행 오차


def test_long_paragraph_split_respects_inline_math():
    sent = "The update is $x_{t+1} = x_t - \\eta. \\nabla f(x_t)$ applied per step. "
    text = "## Method\n\n" + sent * 40
    chunks = MarkdownChunker(max_tokens=50, overlap_tokens=0).chunk(text, doc_id="d", doc_title="T")
    for c in chunks:
        assert c.content.count("$") % 2 == 0


def test_classify_section_priority():
    assert classify_section(["2 Background: Transformer Model"]) == "related_work"
    assert classify_section(["Experiments", "Model Variants"]) == "experiments"
    assert classify_section(["3 Rotary Attention", "3.1 Proposed Approach"]) == "methodology"
    assert classify_section(["A Additional Proofs"]) == "other"
    assert classify_section(["Appendix", "A.1 Proofs"]) == "appendix"
