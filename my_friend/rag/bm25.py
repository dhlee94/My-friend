"""가중 쿼리를 지원하는 BM25 (Okapi, Lucene idf).

rank_bm25 대신 직접 구현한 이유:
  - 쿼리 term별 가중치(subword 0.5, LLM 키워드 1.5)가 필요
  - posting list 기반이라 쿼리당 O(Σ posting 길이) — rank_bm25는 O(N·|q|)
  - 사전 필터 마스크를 그대로 받아 점수 벡터에 적용
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Mapping

import numpy as np


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: Dict[str, tuple] = {}   # term -> (doc_idx int32[], tf float32[])
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.n_docs = 0

    def fit(self, docs: List[List[str]]) -> "BM25":
        raw: Dict[str, tuple] = {}
        for i, toks in enumerate(docs):
            for term, tf in Counter(toks).items():
                raw.setdefault(term, ([], []))
                raw[term][0].append(i)
                raw[term][1].append(tf)
        self.postings = {
            t: (np.asarray(ix, dtype=np.int32), np.asarray(tf, dtype=np.float32)) for t, (ix, tf) in raw.items()
        }
        self.doc_len = np.asarray([len(d) for d in docs], dtype=np.float32)
        self.n_docs = len(docs)
        return self

    @property
    def avgdl(self) -> float:
        return float(self.doc_len.mean()) if self.n_docs else 0.0

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0]) if term in self.postings else 0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: Mapping[str, float]) -> np.ndarray:
        """query: {term: weight} → (N,) float32 점수."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))   # (N,)
        for term, w in query.items():
            post = self.postings.get(term)
            if post is None:
                continue
            ix, tf = post
            out[ix] += w * self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm[ix])
        return out

    # -------------------------------------------------------------- persist
    def state_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_len": self.doc_len.tolist(),
            "postings": {t: [ix.tolist(), tf.tolist()] for t, (ix, tf) in self.postings.items()},
        }

    @classmethod
    def from_state_dict(cls, s: dict) -> "BM25":
        obj = cls(s["k1"], s["b"])
        obj.doc_len = np.asarray(s["doc_len"], dtype=np.float32)
        obj.n_docs = len(obj.doc_len)
        obj.postings = {
            t: (np.asarray(ix, dtype=np.int32), np.asarray(tf, dtype=np.float32)) for t, (ix, tf) in s["postings"].items()
        }
        return obj
