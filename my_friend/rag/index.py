"""하이브리드 인덱스: BM25(+키워드 부스트) ⊕ Dense, RRF 융합, 메타데이터 사전 필터.

벡터 DB 대신 numpy brute-force를 쓰는 이유:
  - 정확 검색(exact kNN). N=100k, D=1024 기준 쿼리당 ~1e8 FLOPs → CPU 수십 ms.
  - 필터를 '검색 전'에 적용 가능. HNSW 계열은 post-filter라 "Methodology만" 같은 좁은 필터에서
    recall이 무너진다.
  - 메모리: N·D·4B = 100k·1024·4 ≈ 400MB. 1M 청크를 넘기면 FAISS IVF/LanceDB로 교체 검토.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .analyzer import EnglishAnalyzer
from .bm25 import BM25
from .embedder import Embedder
from .schema import Chunk, SearchFilter

FORMAT_VERSION = 1


@dataclass
class SearchHit:
    chunk: Chunk
    score: float                 # RRF 융합 점수
    lexical_score: float
    dense_score: float
    lexical_rank: Optional[int]  # 1-based, 후보 밖이면 None
    dense_rank: Optional[int]

    def to_context(self) -> str:
        """에이전트 프롬프트에 주입할 포맷."""
        kws = ", ".join(self.chunk.keyword_weights)
        return f"<source id=\"{self.chunk.chunk_id}\" path=\"{self.chunk.breadcrumb_str}\" keywords=\"{kws}\">\n{self.chunk.content}\n</source>"


class HybridIndex:
    def __init__(
        self,
        chunks: List[Chunk],
        bm25: BM25,
        embeddings: np.ndarray,
        analyzer: EnglishAnalyzer,
        embedder_name: str,
    ) -> None:
        if len(chunks) != embeddings.shape[0] or len(chunks) != bm25.n_docs:
            raise ValueError(
                f"size mismatch: chunks={len(chunks)} embeddings={embeddings.shape} bm25={bm25.n_docs}"
            )
        self.chunks = chunks
        self.bm25 = bm25
        self.embeddings = embeddings          # (N, D) float32, L2 정규화
        self.analyzer = analyzer
        self.embedder_name = embedder_name
        self._build_keyword_postings()

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, chunks: List[Chunk], embedder: Embedder, analyzer: Optional[EnglishAnalyzer] = None) -> "HybridIndex":
        analyzer = analyzer or EnglishAnalyzer()
        texts = [c.indexed_text for c in chunks]
        bm25 = BM25().fit(analyzer.analyze_many(texts))
        embeddings = embedder.embed_documents(texts)            # (N, D)
        return cls(chunks, bm25, embeddings, analyzer, embedder.name)

    def _build_keyword_postings(self) -> None:
        """keyword_weights(표시용 원형) → 분석된 term 기준 역색인. term: [(doc, weight)]"""
        self._kw: Dict[str, List[tuple]] = {}
        for i, c in enumerate(self.chunks):
            per_doc: Dict[str, float] = {}
            for kw, w in c.keyword_weights.items():
                for t in self.analyzer.analyze(kw):
                    if not t.startswith("§"):
                        per_doc[t] = max(per_doc.get(t, 0.0), w)
            for t, w in per_doc.items():
                self._kw.setdefault(t, []).append((i, w))

    # ----------------------------------------------------------------- search
    def filter_mask(self, flt: Optional[SearchFilter]) -> np.ndarray:
        if flt is None or flt.is_empty:
            return np.ones(len(self.chunks), dtype=bool)
        return np.fromiter((flt.matches(c) for c in self.chunks), dtype=bool, count=len(self.chunks))

    def lexical_scores(self, terms: Mapping[str, float], kw_boost: float) -> np.ndarray:
        scores = self.bm25.scores(terms)                         # (N,)
        if kw_boost > 0:
            # 청크 대표 키워드와 쿼리가 겹치면 곱셈 부스트: 본문에 한 번 언급된 청크보다
            # 그 개념이 '주제'인 청크를 위로 올린다. BM25=0인 청크는 부스트돼도 0 유지.
            overlap = np.zeros_like(scores)
            for t, qw in terms.items():
                for i, w in self._kw.get(t, ()):
                    overlap[i] += qw * w
            scores *= 1.0 + kw_boost * overlap
        return scores

    def search(
        self,
        lexical_terms: Mapping[str, float],
        dense_queries: np.ndarray,
        flt: Optional[SearchFilter] = None,
        top_k: int = 5,
        fetch_k: Optional[int] = None,
        rrf_k: int = 60,
        w_lexical: float = 1.0,
        w_dense: float = 1.0,
        kw_boost: float = 0.5,
    ) -> List[SearchHit]:
        """
        lexical_terms : {analyzed_term: weight}
        dense_queries : (Q, D) — 여러 쿼리 벡터(영문 rewrite + 한국어 원문)는 청크별 max로 결합
        """
        n = len(self.chunks)
        if n == 0:
            return []
        if dense_queries.ndim == 1:
            dense_queries = dense_queries[None, :]
        if dense_queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError(f"query dim {dense_queries.shape[1]} != index dim {self.embeddings.shape[1]}")

        cand = np.flatnonzero(self.filter_mask(flt))              # (C,)
        if cand.size == 0:
            return []
        fetch_k = min(fetch_k or max(50, top_k * 10), cand.size)

        lex = self.lexical_scores(lexical_terms, kw_boost)[cand]   # (C,)
        dense = (self.embeddings[cand] @ dense_queries.T).max(axis=1)  # (C,D)@(D,Q) → (C,Q) → (C,)

        lex_rank = _ranks(lex, fetch_k, positive_only=True)        # (C,) 1-based, 0 = 순위권 밖
        dense_rank = _ranks(dense, fetch_k, positive_only=False)

        fused = np.zeros(cand.size, dtype=np.float64)
        fused += np.where(lex_rank > 0, w_lexical / (rrf_k + lex_rank), 0.0)
        fused += np.where(dense_rank > 0, w_dense / (rrf_k + dense_rank), 0.0)

        order = np.argsort(-fused, kind="stable")[:top_k]
        return [
            SearchHit(
                chunk=self.chunks[cand[j]],
                score=float(fused[j]),
                lexical_score=float(lex[j]),
                dense_score=float(dense[j]),
                lexical_rank=int(lex_rank[j]) or None,
                dense_rank=int(dense_rank[j]) or None,
            )
            for j in order
            if fused[j] > 0
        ]

    def neighbors(self, chunk: Chunk, window: int = 1) -> List[Chunk]:
        """같은 문서의 인접 청크 — 에이전트가 '앞뒤 문맥'을 요청할 때."""
        return [
            c for c in self.chunks
            if c.doc_id == chunk.doc_id and 0 < abs(c.order - chunk.order) <= window
        ]

    # ---------------------------------------------------------------- persist
    def save(self, path: str | Path, sources: Optional[Dict[str, str]] = None) -> None:
        """sources: doc_id → 원본 파일 sha256 (pipeline sync가 추가/수정/삭제를 판별하는 데 쓴다)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(p / "embeddings.npy", self.embeddings)
        with open(p / "bm25.json", "w", encoding="utf-8") as f:
            json.dump(self.bm25.state_dict(), f)
        manifest = {
            "format_version": FORMAT_VERSION,
            "embedder": self.embedder_name,
            "dim": int(self.embeddings.shape[1]),
            "n_chunks": len(self.chunks),
            "analyzer": {"encoding": self.analyzer._encoding, "subword_min_len": self.analyzer.subword_min_len,
                         "use_subwords": self.analyzer.use_subwords},
            "sources": sources or {},
        }
        (p / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path: str | Path, expected_embedder: Optional[str] = None) -> "HybridIndex":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        if manifest["format_version"] != FORMAT_VERSION:
            raise ValueError(f"index format {manifest['format_version']} != {FORMAT_VERSION}; rebuild index")
        if expected_embedder and manifest["embedder"] != expected_embedder:
            # 다른 모델의 쿼리 벡터로 검색하면 에러 없이 쓰레기 결과가 나온다 → 여기서 차단
            raise ValueError(f"index built with {manifest['embedder']}, got {expected_embedder}")
        with open(p / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f]
        bm25 = BM25.from_state_dict(json.loads((p / "bm25.json").read_text()))
        embeddings = np.load(p / "embeddings.npy", mmap_mode="r")
        analyzer = EnglishAnalyzer(**manifest["analyzer"])
        return cls(chunks, bm25, np.asarray(embeddings), analyzer, manifest["embedder"])


def read_sources(path: str | Path) -> Dict[str, str]:
    """기존 인덱스의 doc_id → 원본 sha256. 인덱스가 없거나 sources 이전 형식이면 {}."""
    m = Path(path) / "manifest.json"
    return json.loads(m.read_text()).get("sources", {}) if m.exists() else {}


def _ranks(scores: np.ndarray, k: int, positive_only: bool) -> np.ndarray:
    """상위 k개에 1..k 순위, 나머지 0. argpartition으로 O(C) 선택 후 k개만 정렬."""
    ranks = np.zeros(scores.shape[0], dtype=np.int64)
    if k <= 0:
        return ranks
    top = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[0] else np.arange(scores.shape[0])
    top = top[np.argsort(-scores[top], kind="stable")]
    if positive_only:
        top = top[scores[top] > 0]
    ranks[top] = np.arange(1, top.size + 1)
    return ranks
