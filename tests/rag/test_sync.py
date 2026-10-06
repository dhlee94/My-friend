"""pipeline sync (#5): 임베딩만 캐시하고 청킹·키워드 IDF·BM25는 매번 전역 재계산 → 증분 결과 = 전체 재빌드."""
import numpy as np
import pytest

from my_friend.rag.chunker import MarkdownChunker
from my_friend.rag.embed_cache import EmbeddingCache
from my_friend.rag.index import HybridIndex, read_sources
from my_friend.rag.pipeline import build_index, sync_index

from .conftest import PAPER, HashEmbedder

SSM = """# Mamba

## 3 Method

### 3.1 Selective State Spaces

The selection mechanism makes SSM parameters input-dependent. Attention is replaced by a selective scan.

### 3.2 Hardware-aware Algorithm

A parallel scan keeps the recurrent state in SRAM. The decoder uses the same attention-free block.
"""

TCN = """# TCN

## 2 Temporal Convolutional Networks

Causal convolutions ensure no leakage from the future into the past. Dilated convolutions grow the receptive field.
The decoder is a linear layer over the last hidden state.
"""


class CountingEmbedder(HashEmbedder):
    """실제로 임베딩한 텍스트 수를 센다 (캐시 적중이면 호출되지 않는다)."""

    def __init__(self, analyzer, fingerprint=None):
        super().__init__(analyzer)
        self.embedded = 0
        if fingerprint:
            self.cache_fingerprint = fingerprint

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return self._embed(texts)


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "papers"
    (d / "sub").mkdir(parents=True)
    (d / "attention.md").write_text(PAPER)
    (d / "sub" / "mamba.md").write_text(SSM)
    return d


def _sync(corpus, tmp_path, emb, **kw):
    chunker = MarkdownChunker(max_tokens=64, overlap_tokens=8)        # 작은 청크로 문서당 여러 청크
    return sync_index(corpus, tmp_path / "index", emb, tmp_path / "cache.sqlite", chunker=chunker, **kw)


def _same_index(a: HybridIndex, b: HybridIndex):
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]
    assert [c.keyword_weights for c in a.chunks] == [c.keyword_weights for c in b.chunks]
    assert a.bm25.state_dict() == b.bm25.state_dict()
    np.testing.assert_array_equal(np.asarray(a.embeddings), np.asarray(b.embeddings))


def test_first_sync_embeds_everything_and_records_sources(corpus, tmp_path, analyzer):
    emb = CountingEmbedder(analyzer)
    rep = _sync(corpus, tmp_path, emb)
    assert rep.added == ["attention", "mamba"] and not (rep.modified or rep.removed or rep.unchanged)
    assert rep.cache_hits == 0 and rep.cache_misses == emb.embedded == rep.n_chunks > 2
    assert set(read_sources(tmp_path / "index")) == {"attention", "mamba"}
    assert HybridIndex.load(tmp_path / "index").chunks
    assert not (tmp_path / "index.sync-tmp").exists() and not (tmp_path / "index.sync-old").exists()


def test_resync_unchanged_is_free_and_identical(corpus, tmp_path, analyzer):
    _sync(corpus, tmp_path, CountingEmbedder(analyzer))
    before = HybridIndex.load(tmp_path / "index")
    emb = CountingEmbedder(analyzer)
    rep = _sync(corpus, tmp_path, emb)
    assert emb.embedded == 0 and rep.cache_misses == 0 and rep.cache_hits == rep.n_chunks
    assert rep.unchanged == ["attention", "mamba"]
    _same_index(before, HybridIndex.load(tmp_path / "index"))


def test_add_embeds_only_new_doc_and_matches_full_rebuild(corpus, tmp_path, analyzer):
    _sync(corpus, tmp_path, CountingEmbedder(analyzer))
    (corpus / "tcn.md").write_text(TCN)
    emb = CountingEmbedder(analyzer)
    rep = _sync(corpus, tmp_path, emb)
    synced = HybridIndex.load(tmp_path / "index")
    new_chunks = sum(c.doc_id == "tcn" for c in synced.chunks)
    assert rep.added == ["tcn"] and emb.embedded == new_chunks == rep.cache_misses
    # IDF drift 없음: 기존 문서의 키워드 가중치까지 처음부터 전체 빌드한 결과와 동일
    fresh = build_index(sorted(corpus.rglob("*.md")), HashEmbedder(analyzer), MarkdownChunker(64, 8))
    _same_index(synced, fresh)


def test_modify_reembeds_only_changed_chunks(corpus, tmp_path, analyzer):
    _sync(corpus, tmp_path, CountingEmbedder(analyzer))
    (corpus / "sub" / "mamba.md").write_text(SSM.replace("keeps the recurrent state in SRAM", "fuses kernels in SRAM"))
    emb = CountingEmbedder(analyzer)
    rep = _sync(corpus, tmp_path, emb)
    mamba_chunks = sum(c.doc_id == "mamba" for c in HybridIndex.load(tmp_path / "index").chunks)
    assert rep.modified == ["mamba"] and rep.unchanged == ["attention"]
    assert 1 <= emb.embedded < mamba_chunks + 1 and rep.cache_hits > 0


def test_delete_removes_doc_and_gc_drops_its_vectors(corpus, tmp_path, analyzer):
    _sync(corpus, tmp_path, CountingEmbedder(analyzer))
    n_all = len(EmbeddingCache(tmp_path / "cache.sqlite"))
    (corpus / "sub" / "mamba.md").unlink()
    rep = _sync(corpus, tmp_path, CountingEmbedder(analyzer))
    idx = HybridIndex.load(tmp_path / "index")
    assert rep.removed == ["mamba"] and all(c.doc_id == "attention" for c in idx.chunks)
    assert len(EmbeddingCache(tmp_path / "cache.sqlite")) == n_all          # gc 없이는 보관 (재추가 시 무료)

    rep = _sync(corpus, tmp_path, CountingEmbedder(analyzer), gc=True)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    assert rep.gc_deleted == n_all - len(idx.chunks) > 0 and len(cache) == len(idx.chunks)


def test_embedder_fingerprint_change_invalidates_cache(corpus, tmp_path, analyzer):
    _sync(corpus, tmp_path, CountingEmbedder(analyzer, fingerprint="test/hash-256|max_seq_length=1024"))
    emb = CountingEmbedder(analyzer, fingerprint="test/hash-256|max_seq_length=2048")
    rep = _sync(corpus, tmp_path, emb)
    assert rep.cache_hits == 0 and emb.embedded == rep.n_chunks


def test_failed_sync_keeps_previous_index(corpus, tmp_path, analyzer):
    _sync(corpus, tmp_path, CountingEmbedder(analyzer))
    before = HybridIndex.load(tmp_path / "index")
    (corpus / "tcn.md").write_text(TCN)

    class Boom(CountingEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("GPU fell over")

    with pytest.raises(RuntimeError):
        _sync(corpus, tmp_path, Boom(analyzer))
    _same_index(before, HybridIndex.load(tmp_path / "index"))


def test_duplicate_doc_ids_rejected(corpus, tmp_path, analyzer):
    (corpus / "sub" / "attention.md").write_text(SSM)
    with pytest.raises(ValueError, match="duplicate doc id 'attention'"):
        _sync(corpus, tmp_path, CountingEmbedder(analyzer))
