"""청크별 Top-K 영문 키워드 + 가중치 추출 (코퍼스 TF-IDF, 의존성 없음).

- 후보: 불용어로 끊기는 연속 구간의 unigram + bigram ("multi head", "positional encoding").
  stem 단위로 집계하고, 표시는 가장 빈번한 원형(surface form)으로 복원.
- 점수: (1 + log tf) * idf * boost,  boost = bigram 1.2 / breadcrumb 등장 1.3.
- 가중치: 청크 내 max로 정규화 → [0, 1]. 에이전트가 읽기 쉽고 청크 간 비교도 가능.

주의: fit()이 코퍼스 전체 df를 쓰므로 문서 추가 시 기존 청크 키워드는 재계산해야 정확하다
(증분 인덱싱 시 idf drift).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from .analyzer import STOPWORDS, EnglishAnalyzer
from .schema import Chunk

Key = Tuple[str, ...]  # stem tuple (len 1 or 2)


class KeywordExtractor:
    def __init__(
        self,
        analyzer: EnglishAnalyzer,
        top_k: int = 8,
        max_df_ratio: float = 0.5,
        bigram_boost: float = 1.2,
        header_boost: float = 1.3,
    ) -> None:
        self.analyzer = analyzer
        self.top_k = top_k
        self.max_df_ratio = max_df_ratio
        self.bigram_boost = bigram_boost
        self.header_boost = header_boost

    def _candidates(self, text: str) -> Tuple[Counter, Dict[Key, Counter]]:
        tf: Counter = Counter()
        surface: Dict[Key, Counter] = defaultdict(Counter)
        run: List[Tuple[str, str]] = []   # (stem, surface) — 불용어/숫자에서 끊김

        def close_run() -> None:
            for i, (s, w) in enumerate(run):
                tf[(s,)] += 1
                surface[(s,)][w] += 1
                if i + 1 < len(run):
                    s2, w2 = run[i + 1]
                    tf[(s, s2)] += 1
                    surface[(s, s2)][f"{w} {w2}"] += 1
            run.clear()

        for raw in self.analyzer.words(text):
            low = raw.lower()
            if low in STOPWORDS or len(low) < 3 or not any(c.isalpha() for c in low):
                close_run()
                continue
            run.append((self.analyzer.stem(low), low))
        close_run()
        return tf, surface

    def fit_transform(self, chunks: Sequence[Chunk]) -> List[Dict[str, float]]:
        per_chunk = [self._candidates(c.content) for c in chunks]
        n = len(chunks)
        df: Counter = Counter()
        for tf, _ in per_chunk:
            df.update(tf.keys())
        # 코퍼스가 작을 땐 max_df 컷이 오히려 핵심어를 날리므로 n ≥ 20에서만 적용
        max_df = self.max_df_ratio * n if n >= 20 else float("inf")

        results: List[Dict[str, float]] = []
        for chunk, (tf, surface) in zip(chunks, per_chunk):
            header_stems = {self.analyzer.stem(w.lower()) for w in self.analyzer.words(" ".join(chunk.breadcrumb))}
            scored: List[Tuple[float, Key]] = []
            for key, f in tf.items():
                if df[key] > max_df:
                    continue
                if len(key) == 2 and f < 2:        # 1회 등장 bigram은 대부분 우연한 인접
                    continue
                idf = math.log((n + 1) / (df[key] + 1)) + 1.0
                score = (1.0 + math.log(f)) * idf
                if len(key) == 2:
                    score *= self.bigram_boost
                if header_stems.intersection(key):
                    score *= self.header_boost
                scored.append((score, key))
            scored.sort(key=lambda x: (-x[0], x[1]))

            picked: Dict[str, float] = {}
            covered: set = set()
            for score, key in scored:
                # bigram이 이미 뽑혔으면 그 구성 unigram은 중복이므로 스킵
                if len(key) == 1 and key[0] in covered:
                    continue
                picked[surface[key].most_common(1)[0][0]] = score
                covered.update(key)
                if len(picked) >= self.top_k:
                    break
            top = max(picked.values(), default=1.0)
            results.append({k: round(v / top, 4) for k, v in picked.items()})
        return results

    def assign(self, chunks: Sequence[Chunk]) -> None:
        for chunk, kw in zip(chunks, self.fit_transform(chunks)):
            chunk.keyword_weights = kw
