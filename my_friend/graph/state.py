"""그래프 상태.

수명 주기가 다른 두 종류의 필드가 섞여 있다:
  - 세션 지속(체크포인터가 thread_id 단위로 보존): messages, files, entry_file, spec, scout
    → 다음 턴의 sparring 부분 수정이 이전 full build 결과 위에서 동작한다.
  - 턴 단위(turn_start가 매 턴 초기화): change, review, validation, reflection, iteration, sparring
"""
from __future__ import annotations

import operator
from typing import Dict, List, Optional

from typing_extensions import Annotated, TypedDict

from my_friend.agents.schemas import (
    ChangeRequest,
    Mode,
    ModelSpec,
    Reflection,
    ReviewDecision,
    ScoutReport,
    SparringTurn,
    ValidationReport,
)


class Turn(TypedDict):
    role: str       # "user" | "assistant"
    name: str       # 발화 노드 (user / paper_scout / sparring_partner / ...)
    content: str


def merge_files(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    """파일 단위 upsert. 빈 문자열은 삭제 의미."""
    merged = dict(old or {})
    for path, src in (new or {}).items():
        if src == "":
            merged.pop(path, None)
        else:
            merged[path] = src
    return merged


class GraphState(TypedDict, total=False):
    # --- input
    request: str
    requested_mode: Optional[Mode]          # 사용자가 모드를 강제할 때

    # --- session (persisted across turns)
    messages: Annotated[List[Turn], operator.add]
    files: Annotated[Dict[str, str], merge_files]
    entry_file: Optional[str]
    spec: Optional[ModelSpec]
    scout: Optional[ScoutReport]

    # --- per turn
    mode: Mode
    route_reason: str
    sparring: Optional[SparringTurn]
    change: Optional[ChangeRequest]
    review: Optional[ReviewDecision]
    validation: Optional[ValidationReport]
    reflection: Optional[Reflection]
    iteration: int                          # validator→reflector→patch 루프 횟수
    max_iterations: int


PER_TURN_RESET: GraphState = {
    "sparring": None,
    "change": None,
    "review": None,
    "validation": None,
    "reflection": None,
    "iteration": 0,
}
