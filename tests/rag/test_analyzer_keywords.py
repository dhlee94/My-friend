from my_friend.rag.analyzer import SUBWORD_PREFIX
from my_friend.rag.chunker import MarkdownChunker
from my_friend.rag.keywords import KeywordExtractor

from .conftest import PAPER


def test_stemming_and_stopwords(analyzer):
    toks = analyzer.analyze("The attentions are attending")
    assert "the" not in toks and "are" not in toks
    assert toks.count(analyzer.stem("attention")) == 1


def test_identifier_split(analyzer):
    toks = analyzer.analyze("nn.MultiheadAttention")
    assert "nn.multiheadattention" in toks
    assert analyzer.stem("attention") in toks


def test_subwords_link_spelling_variants(analyzer):
    a = {t for t in analyzer.analyze("RMSNorm") if t.startswith(SUBWORD_PREFIX)}
    b = {t for t in analyzer.analyze("rms_norm") if t.startswith(SUBWORD_PREFIX)}
    assert a and a & b


def test_latex_commands_dropped(analyzer):
    toks = analyzer.analyze(r"\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)")
    assert "frac" not in toks and "mathrm" not in toks
    assert "softmax" in toks


def test_query_terms_weights(analyzer):
    terms = analyzer.query_terms("RMSNorm layer", weight=1.5)
    assert terms["rmsnorm"] == 1.5
    assert all(w == 0.75 for t, w in terms.items() if t.startswith(SUBWORD_PREFIX))


def test_keyword_weights(analyzer):
    chunks = MarkdownChunker().chunk(PAPER, doc_id="d")
    KeywordExtractor(analyzer, top_k=5).assign(chunks)
    for c in chunks:
        assert 0 < len(c.keyword_weights) <= 5
        assert max(c.keyword_weights.values()) == 1.0
        assert all(0 < w <= 1.0 for w in c.keyword_weights.values())
    pe = next(c for c in chunks if c.breadcrumb[-1] == "3.5 Positional Encoding")
    assert any("position" in k for k in pe.keyword_weights)
