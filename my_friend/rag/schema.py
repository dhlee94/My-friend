from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, FrozenSet, List, Optional

# 정규화된 섹션 라벨. 에이전트/LLM 필터는 이 값만 사용한다.
SECTION_TYPES = (
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "experiments",
    "results",
    "conclusion",
    "appendix",
    "references",
    "other",
)

# 순서가 곧 우선순위: "Background: Transformer Model"이 methodology가 아니라 related_work로 가도록
# 구체적/배타적인 패턴을 먼저 둔다.
_SECTION_PATTERNS = [
    ("abstract", r"abstract"),
    ("references", r"references|bibliography"),
    ("appendix", r"appendix|appendices|supplementary"),
    ("related_work", r"related\s+work|background|preliminar(?:y|ies)|prior\s+work"),
    ("introduction", r"introduction|motivation"),
    ("conclusion", r"conclusions?|future\s+work|limitations?|concluding"),
    ("experiments", r"experiments?|experimental|evaluation|setup|implementation\s+details|training\s+details|benchmarks?|ablations?"),
    ("results", r"results?|analysis|discussion"),
    ("methodology", r"methods?|methodology|approach|model|architecture|framework|proposed|algorithm|formulation|design"),
]
_SECTION_RE = [(label, re.compile(rf"\b(?:{pat})\b", re.I)) for label, pat in _SECTION_PATTERNS]
# "3.2 ", "III. ", "A.1 " 같은 번호 접두사 제거
_NUMBERING_RE = re.compile(r"^\s*(?:[0-9]+(?:\.[0-9]+)*|[IVXLC]+|[A-Z](?:\.[0-9]+)*)[.)]?\s+")


def classify_section(breadcrumb: List[str]) -> str:
    """상위 헤더부터 매칭 — 'Experiments > Attention Variants'는 experiments.

    상위 헤더가 'Scaled Rotary Attention'처럼 비표준 이름이면 하위로 내려가며 탐색하고,
    끝까지 없으면 'other'.
    """
    for title in breadcrumb:
        title = _NUMBERING_RE.sub("", title)
        for label, regex in _SECTION_RE:
            if regex.search(title):
                return label
    return "other"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    order: int                      # 문서 내 순서 (인접 청크 확장용)
    breadcrumb: List[str]           # H1~H3 경로, e.g. ["3 Model Architecture", "3.2 Attention"]
    section_type: str               # classify_section(breadcrumb)
    content: str                    # 원본 마크다운 본문 (헤더 제외)
    block_types: List[str]          # 포함된 블록 종류: paragraph/equation/table/code/list
    token_count: int
    oversized: bool = False         # 보호 블록 하나가 max_tokens를 초과해 단독 청크가 된 경우
    keyword_weights: Dict[str, float] = field(default_factory=dict)

    @property
    def breadcrumb_str(self) -> str:
        return " > ".join([self.doc_title, *self.breadcrumb])

    @property
    def indexed_text(self) -> str:
        """BM25/Dense 양쪽에 들어가는 텍스트. 헤더 경로를 본문에 병합해
        'Attention' 같은 짧은 청크도 섹션 문맥으로 검색되게 한다."""
        return f"[{self.breadcrumb_str}]\n{self.content}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


@dataclass
class SearchFilter:
    """검색 전(pre-filter)에 적용되는 메타데이터 필터. 모든 조건은 AND."""

    section_types: Optional[FrozenSet[str]] = None      # {"methodology"}
    breadcrumb_contains: Optional[str] = None           # 대소문자 무시 부분일치, e.g. "positional encoding"
    doc_ids: Optional[FrozenSet[str]] = None
    require_blocks: Optional[FrozenSet[str]] = None     # {"equation"} → 수식 포함 청크만

    def __post_init__(self) -> None:
        if self.section_types:
            unknown = set(self.section_types) - set(SECTION_TYPES)
            if unknown:
                raise ValueError(f"unknown section types: {unknown}")

    def matches(self, chunk: Chunk) -> bool:
        if self.section_types and chunk.section_type not in self.section_types:
            return False
        if self.doc_ids and chunk.doc_id not in self.doc_ids:
            return False
        if self.breadcrumb_contains and self.breadcrumb_contains.lower() not in chunk.breadcrumb_str.lower():
            return False
        if self.require_blocks and not self.require_blocks.issubset(chunk.block_types):
            return False
        return True

    @property
    def is_empty(self) -> bool:
        return not (self.section_types or self.breadcrumb_contains or self.doc_ids or self.require_blocks)
