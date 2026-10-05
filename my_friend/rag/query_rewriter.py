"""교차 언어 쿼리 재작성: 한국어 질문 → 영문 검색 쿼리 + 키워드 + 섹션 필터.

한글이 없는 쿼리는 LLM을 건너뛴다 (지연/비용 0). 결과는 프로세스 내 LRU 캐시 —
에이전트 루프에서 같은 질의가 반복 호출되는 경우가 많다.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

from my_friend.core.llm import StructuredLLM

from .schema import SECTION_TYPES

_HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")

SYSTEM_PROMPT = """You convert a Korean (or mixed Korean/English) question about machine-learning papers and \
PyTorch code into an English search request for a hybrid BM25 + dense index of English papers and code.

- english_query: one natural English sentence a paper author would write, using standard ML terminology \
(e.g. "어텐션 스코어 스케일링" -> "scaled dot-product attention scaling by sqrt(d_k)").
- keywords: 3-8 high-precision English terms. Include canonical names, common abbreviations and their \
expansions (e.g. "RoPE", "rotary position embedding"), and PyTorch identifiers when code is implied \
(e.g. "nn.MultiheadAttention", "F.scaled_dot_product_attention"). Do not include generic words.
- section_filter: set only when the user explicitly restricts where to search \
(e.g. "방법론 섹션에서만" -> "methodology", "실험 부분" -> "experiments"). Otherwise "any".
Keep identifiers, equations and English terms that already appear in the question verbatim."""

SCHEMA = {
    "type": "object",
    "properties": {
        "english_query": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "section_filter": {"type": "string", "enum": ["any", *[s for s in SECTION_TYPES if s != "other"]]},
    },
    "required": ["english_query", "keywords", "section_filter"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RewrittenQuery:
    original: str
    english_query: str
    keywords: List[str] = field(default_factory=list)
    section_filter: Optional[str] = None
    rewritten: bool = False


def contains_hangul(text: str) -> bool:
    return bool(_HANGUL.search(text))


class QueryRewriter:
    def __init__(self, llm: StructuredLLM, cache_size: int = 256) -> None:
        self.llm = llm
        self._cache: "OrderedDict[str, RewrittenQuery]" = OrderedDict()
        self._cache_size = cache_size

    def rewrite(self, query: str) -> RewrittenQuery:
        query = query.strip()
        if not contains_hangul(query):
            return RewrittenQuery(original=query, english_query=query)
        if query in self._cache:
            self._cache.move_to_end(query)
            return self._cache[query]

        out = self.llm.generate_json(SYSTEM_PROMPT, query, SCHEMA)
        section = out.get("section_filter")
        result = RewrittenQuery(
            original=query,
            english_query=out["english_query"].strip() or query,
            keywords=[k.strip() for k in out.get("keywords", []) if k.strip()][:8],
            section_filter=section if section in SECTION_TYPES else None,
            rewritten=True,
        )
        self._cache[query] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result
