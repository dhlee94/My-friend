"""영문 BM25용 분석기: Snowball 어근 + 식별자 분해 + tiktoken subword.

토큰 3계층 (가중치는 쿼리 측에서 적용):
  1. stem      : "attentions" → "attent"                                    (w=1.0)
  2. exact id  : "nn.MultiheadAttention" → "nn.multiheadattention"         (w=1.0)
                 + 분해 조각 stem: "nn", "multihead", "attent"
  3. subword   : OOV/합성어를 BPE 조각으로 → "§rms", "§norm"                (w=0.5)
     → "RMSNorm" ↔ "rms_norm" ↔ "RMS normalization" 이 부분 일치로 연결된다.

subword는 길이 ≥ 6 & BPE 조각 ≥ 2인 토큰에만 적용 — 모든 단어에 붙이면 인덱스 크기가 ~2배가 되고
흔한 조각("tion" 등)이 BM25 노이즈가 된다.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from nltk.stem.snowball import SnowballStemmer

SUBWORD_PREFIX = "§"

# nltk stopwords 코퍼스는 별도 다운로드가 필요해 오프라인 환경을 위해 내장.
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your yours yourself yourselves
    also via et al eg ie fig figure table eq equation section use used using we our thus however therefore""".split()
)

_LATEX_CMD = re.compile(r"\\[A-Za-z]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_.\-][A-Za-z0-9]+)*|\d+(?:\.\d+)?")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_SPLIT_ID = re.compile(r"[_.\-]")


def _is_identifier(tok: str) -> bool:
    return bool(_SPLIT_ID.search(tok)) or (tok != tok.lower() and tok != tok.upper() and not tok.istitle()) or (
        any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)
    )


class EnglishAnalyzer:
    def __init__(self, encoding: str = "cl100k_base", subword_min_len: int = 6, use_subwords: bool = True) -> None:
        self._stemmer = SnowballStemmer("english")
        self.subword_min_len = subword_min_len
        self.use_subwords = use_subwords
        self._enc = None
        self._encoding = encoding
        # 인스턴스별 캐시 (stem 호출이 인덱싱 hot path)
        self.stem = lru_cache(maxsize=200_000)(self._stem)
        self._subwords = lru_cache(maxsize=200_000)(self._subwords_uncached)

    # tiktoken 인코더 로드는 첫 사용 시점으로 지연 (pickle/오프라인 import 친화)
    @property
    def enc(self):
        if self._enc is None:
            import tiktoken

            self._enc = tiktoken.get_encoding(self._encoding)
        return self._enc

    def _stem(self, word: str) -> str:
        return self._stemmer.stem(word)

    def _subwords_uncached(self, word: str) -> tuple:
        ids = self.enc.encode(" " + word, disallowed_special=())
        if len(ids) < 2:
            return ()
        pieces = (self.enc.decode([i]).strip().lower() for i in ids)
        return tuple(SUBWORD_PREFIX + p for p in pieces if len(p) >= 2 and p.isalnum())

    def words(self, text: str) -> List[str]:
        """원형 토큰 (LaTeX 명령어 제거). 키워드 추출기가 surface form 복원에 사용."""
        return _WORD.findall(_LATEX_CMD.sub(" ", text))

    def analyze_word(self, raw: str) -> List[str]:
        low = raw.lower()
        if low in STOPWORDS or (len(low) < 2 and not low.isdigit()):
            return []
        out: List[str] = []
        if _is_identifier(raw):
            out.append(low)
            parts = [p for seg in _SPLIT_ID.split(raw) for p in _CAMEL.findall(seg)]
            for p in parts:
                pl = p.lower()
                if len(pl) >= 2 and pl not in STOPWORDS and pl != low:
                    out.append(self.stem(pl))
        else:
            out.append(self.stem(low))
        if self.use_subwords and len(low) >= self.subword_min_len and not low.isdigit():
            compact = _SPLIT_ID.sub("", low)
            out.extend(self._subwords(compact))
        return out

    def analyze(self, text: str) -> List[str]:
        out: List[str] = []
        for w in self.words(text):
            out.extend(self.analyze_word(w))
        return out

    def query_terms(self, text: str, weight: float = 1.0, subword_weight: float = 0.5,
                    acc: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """쿼리 → {term: weight}. 같은 term이 여러 번 나오면 max (쿼리 길이로 점수가 부풀지 않게)."""
        acc = {} if acc is None else acc
        for t in self.analyze(text):
            w = weight * (subword_weight if t.startswith(SUBWORD_PREFIX) else 1.0)
            acc[t] = max(acc.get(t, 0.0), w)
        return acc

    def analyze_many(self, texts: Iterable[str]) -> List[List[str]]:
        return [self.analyze(t) for t in texts]
