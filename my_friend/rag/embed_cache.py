"""로컬 임베딩 캐시 (SQLite, 표준 라이브러리만).

키 = sha256(임베더 지문 + 문서 텍스트). 지문에는 모델 이름뿐 아니라 max_seq_length·doc_prefix가 들어간다:
같은 텍스트라도 이 값이 바뀌면 벡터가 달라지므로, 텍스트만 키로 쓰면 옛 벡터가 조용히 재사용된다.

`pipeline sync`는 매번 코퍼스 전체를 다시 청킹하고 키워드/BM25를 전역 재계산한다 (밀리초 단위).
비싼 임베딩만 이 캐시로 건너뛰므로 증분 추가/수정/삭제가 전체 재빌드와 같은 결과를 낸다 (IDF drift 없음, #5).
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

from .embedder import Embedder


def embedder_fingerprint(embedder: Embedder) -> str:
    return getattr(embedder, "cache_fingerprint", None) or embedder.name


def text_key(fingerprint: str, text: str) -> str:
    return hashlib.sha256(f"{fingerprint}\0{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )

    def get_many(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        uniq = list(dict.fromkeys(keys))
        for i in range(0, len(uniq), 500):                      # SQLite 변수 개수 제한
            batch = uniq[i : i + 500]
            rows = self.conn.execute(
                f"SELECT key, dim, vec FROM embeddings WHERE key IN ({','.join('?' * len(batch))})", batch
            )
            for key, dim, blob in rows:
                out[key] = np.frombuffer(blob, dtype=np.float32, count=dim)
        return out

    def put_many(self, items: Iterable[tuple]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)",
                ((k, int(v.shape[0]), np.ascontiguousarray(v, dtype=np.float32).tobytes()) for k, v in items),
            )

    def keys(self) -> Set[str]:
        return {k for (k,) in self.conn.execute("SELECT key FROM embeddings")}

    def gc(self, keep: Set[str]) -> int:
        """keep에 없는 벡터 삭제 (현재 인덱스가 참조하지 않는 것). 삭제 개수를 돌려준다."""
        stale = list(self.keys() - keep)
        with self.conn:
            for i in range(0, len(stale), 500):
                batch = stale[i : i + 500]
                self.conn.execute(f"DELETE FROM embeddings WHERE key IN ({','.join('?' * len(batch))})", batch)
        if stale:
            self.conn.execute("VACUUM")
        return len(stale)

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def close(self) -> None:
        self.conn.close()


class CachedEmbedder:
    """embed_documents만 캐시한다. 쿼리 임베딩은 매번 달라서 캐시 이득이 없다."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self.inner, self.cache = inner, cache
        self.name = inner.name
        self.fingerprint = embedder_fingerprint(inner)
        self.hits = self.misses = 0
        self.last_keys: List[str] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        keys = [text_key(self.fingerprint, t) for t in texts]
        self.last_keys = keys
        found = self.cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in found]
        # 같은 텍스트가 여러 번 나와도 한 번만 임베딩
        todo = list(dict.fromkeys(keys[i] for i in missing))
        text_of = {keys[i]: texts[i] for i in missing}
        if todo:
            vecs = self.inner.embed_documents([text_of[k] for k in todo])
            new = dict(zip(todo, vecs))
            self.cache.put_many(new.items())
            found.update(new)
        self.hits += len(keys) - len(missing)
        self.misses += len(todo)
        return np.stack([found[k] for k in keys]).astype(np.float32) if keys else np.zeros((0, 0), np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.inner.embed_queries(texts)
