"""에이전트용 검색 인터페이스: rewrite → (lexical terms, dense vectors, filter) → HybridIndex.search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .embedder import Embedder
from .index import HybridIndex, SearchHit
from .query_rewriter import QueryRewriter, RewrittenQuery
from .schema import SearchFilter


@dataclass
class RetrievalResult:
    query: RewrittenQuery
    hits: List[SearchHit]
    applied_filter: Optional[SearchFilter]
    notes: List[str] = field(default_factory=list)

    def to_context(self) -> str:
        return "\n\n".join(h.to_context() for h in self.hits)


class HybridRetriever:
    def __init__(
        self,
        index: HybridIndex,
        embedder: Embedder,
        rewriter: Optional[QueryRewriter] = None,
        keyword_weight: float = 1.5,
        dense_with_original: bool = True,
    ) -> None:
        if index.embedder_name != embedder.name:
            raise ValueError(f"index embedder {index.embedder_name} != query embedder {embedder.name}")
        self.index = index
        self.embedder = embedder
        self.rewriter = rewriter
        self.keyword_weight = keyword_weight
        self.dense_with_original = dense_with_original

    def retrieve(self, query: str, top_k: int = 5, flt: Optional[SearchFilter] = None, **search_kwargs) -> RetrievalResult:
        """
        flt가 명시되면(에이전트가 "Methodology에서만" 같은 툴 인자로 지정) 그대로 쓰고,
        없으면 LLM이 질문에서 추론한 section_filter를 사용한다.
        추론된 필터로 0건이면 필터를 풀고 재검색 — 명시 필터는 절대 풀지 않는다.
        """
        rq = self.rewriter.rewrite(query) if self.rewriter else RewrittenQuery(original=query, english_query=query)
        notes: List[str] = []

        inferred = False
        if flt is None and rq.section_filter:
            flt = SearchFilter(section_types=frozenset({rq.section_filter}))
            inferred = True

        # BM25: 영문 쿼리(w=1) + LLM 키워드(w=1.5). 한국어 원문은 영문 분석기에서 어차피 토큰이 안 나옴.
        terms = self.index.analyzer.query_terms(rq.english_query)
        for kw in rq.keywords:
            self.index.analyzer.query_terms(kw, weight=self.keyword_weight, acc=terms)

        # Dense: 영문 rewrite + (rewrite된 경우) 한국어 원문. 다국어 임베더라 원문도 유효한 신호이고,
        # LLM이 의도를 잘못 번역했을 때의 안전망이 된다.
        dense_texts = [rq.english_query]
        if rq.rewritten and self.dense_with_original:
            dense_texts.append(rq.original)
        q_vecs: np.ndarray = self.embedder.embed_queries(dense_texts)   # (Q, D)

        hits = self.index.search(terms, q_vecs, flt=flt, top_k=top_k, **search_kwargs)
        if not hits and inferred:
            notes.append(f"no hits in inferred section '{rq.section_filter}', retried without filter")
            flt = None
            hits = self.index.search(terms, q_vecs, flt=None, top_k=top_k, **search_kwargs)
        return RetrievalResult(query=rq, hits=hits, applied_filter=flt, notes=notes)
