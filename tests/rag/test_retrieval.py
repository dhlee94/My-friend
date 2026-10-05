import numpy as np
import pytest

from my_friend.rag.index import HybridIndex
from my_friend.rag.pipeline import build_index
from my_friend.rag.query_rewriter import QueryRewriter
from my_friend.rag.retriever import HybridRetriever
from my_friend.rag.schema import SearchFilter

from .conftest import PAPER, FakeLLM


@pytest.fixture
def index(tmp_path, hash_embedder, analyzer):
    p = tmp_path / "vaswani17.md"
    p.write_text(PAPER)
    return build_index([p], hash_embedder, analyzer=analyzer)


def test_korean_query_is_rewritten_and_hits_methodology(index, hash_embedder):
    llm = FakeLLM({
        "english_query": "scaled dot-product attention scaling factor sqrt(d_k)",
        "keywords": ["scaled dot-product attention", "softmax", "d_k"],
        "section_filter": "methodology",
    })
    retriever = HybridRetriever(index, hash_embedder, QueryRewriter(llm))
    res = retriever.retrieve("방법론 섹션에서만 어텐션 스케일링 이유 찾아줘", top_k=3)

    assert res.query.rewritten and llm.calls
    assert res.applied_filter.section_types == frozenset({"methodology"})
    assert all(h.chunk.section_type == "methodology" for h in res.hits)
    assert "3.2 Scaled Dot-Product Attention" in res.hits[0].chunk.breadcrumb
    assert res.hits[0].lexical_rank == 1


def test_english_query_skips_llm(index, hash_embedder):
    llm = FakeLLM({})
    retriever = HybridRetriever(index, hash_embedder, QueryRewriter(llm))
    res = retriever.retrieve("positional encoding sine cosine", top_k=2)
    assert not llm.calls
    assert res.hits[0].chunk.breadcrumb[-1] == "3.5 Positional Encoding"


def test_rewrite_is_cached(index, hash_embedder):
    llm = FakeLLM({"english_query": "BLEU", "keywords": [], "section_filter": "any"})
    retriever = HybridRetriever(index, hash_embedder, QueryRewriter(llm))
    retriever.retrieve("BLEU 점수 표")
    retriever.retrieve("BLEU 점수 표")
    assert len(llm.calls) == 1


def test_explicit_filter_and_breadcrumb_filter(index, hash_embedder):
    retriever = HybridRetriever(index, hash_embedder)
    res = retriever.retrieve("BLEU", flt=SearchFilter(require_blocks=frozenset({"table"})))
    assert res.hits and all("table" in h.chunk.block_types for h in res.hits)
    res = retriever.retrieve("attention", flt=SearchFilter(breadcrumb_contains="positional"))
    assert all("Positional" in h.chunk.breadcrumb_str for h in res.hits)


def test_inferred_filter_falls_back_when_empty(index, hash_embedder):
    llm = FakeLLM({"english_query": "attention", "keywords": [], "section_filter": "conclusion"})
    res = HybridRetriever(index, hash_embedder, QueryRewriter(llm)).retrieve("결론에서 어텐션")
    assert res.hits and res.applied_filter is None and res.notes


def test_explicit_filter_never_relaxed(index, hash_embedder):
    res = HybridRetriever(index, hash_embedder).retrieve(
        "attention", flt=SearchFilter(section_types=frozenset({"conclusion"}))
    )
    assert res.hits == []


def test_keyword_boost_reorders(index):
    terms = index.analyzer.query_terms("positional encoding")
    base = index.lexical_scores(terms, kw_boost=0.0)
    boosted = index.lexical_scores(terms, kw_boost=0.5)
    assert np.all(boosted >= base)
    assert np.any(boosted > base)


def test_save_load_roundtrip(index, tmp_path, hash_embedder):
    out = tmp_path / "idx"
    index.save(out)
    loaded = HybridIndex.load(out, expected_embedder=hash_embedder.name)
    q = hash_embedder.embed_queries(["softmax attention"])
    terms = index.analyzer.query_terms("softmax attention")
    a = [h.chunk.chunk_id for h in index.search(terms, q, top_k=3)]
    b = [h.chunk.chunk_id for h in loaded.search(terms, q, top_k=3)]
    assert a == b
    assert loaded.chunks[0].keyword_weights == index.chunks[0].keyword_weights
    with pytest.raises(ValueError):
        HybridIndex.load(out, expected_embedder="BAAI/bge-m3")


def test_dim_mismatch_raises(index):
    with pytest.raises(ValueError):
        index.search({}, np.zeros((1, 7), dtype=np.float32))
