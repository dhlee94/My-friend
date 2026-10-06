"""Dense 임베더.

기본값 BAAI/bge-m3 선택 근거:
  - 다국어 공유 공간 → 한국어 원문 쿼리로도 영문 청크 검색 가능 (rewrite 실패 시 안전망)
  - 8192 토큰 컨텍스트 → oversized 표/수식 청크도 잘리지 않음
  - prefix 불필요 (e5 계열은 "query: "/"passage: " 누락 시 성능 급락)
비용: XLM-R large(568M). MPS에서 청크 1만 개 인덱싱 수 분 수준, fp16은 cuda에서만 사용.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import numpy as np

from my_friend.core.device import resolve_device


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """(N, D) float32, L2 정규화."""
        ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """(Q, D) float32, L2 정규화."""
        ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 16,
        query_prefix: str = "",
        doc_prefix: str = "",
        max_seq_length: Optional[int] = 2048,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.device = resolve_device(device)
        kwargs = {"torch_dtype": "float16"} if self.device == "cuda" else {}
        self.model = SentenceTransformer(model_name, device=self.device, model_kwargs=kwargs)
        if max_seq_length:
            # attention O(L²) → 캡으로 패딩 낭비/메모리 억제. 768(tiktoken) 청크는 bge 토큰으로 p95≈1.1k라
            # 1024 캡이면 7%가 잘린다 → 2048 (잘림은 쪼갤 수 없는 oversized 블록만 남음, 임베딩 시간 +27%)
            self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) > 256
        )
        return emb.astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])
