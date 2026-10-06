"""그래프 조립.

    START → turn_start → paper_scout → router ─┬─ sparring ──→ sparring_partner ─┬─ (너무 복잡) escalate ──→ planner
                                               │                                  └→ human_review ⟲(revise)
                                               │                                       ├ approve  → partial_executor ─┐
                                               │                                       └ escalate → planner           │
                                               └─ full_build → planner → human_review ⟲(revise)                       │
                                                                            └ approve → executor ─────────────────────┤
                                                                                                                      ▼
                     수렴점(두 트랙 공통): validator → reflector ─┬ patch    → partial_executor → validator …
                                                                   ├ ask_user → human_review
                                                                   └ accept / budget 초과 → END
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from my_friend.agents.base import AgentDeps
from my_friend.agents.executor import Executor, PartialExecutor
from my_friend.agents.human_review import human_review
from my_friend.agents.paper_scout import PaperScout
from my_friend.agents.planner import Planner
from my_friend.agents.reflector import Reflector
from my_friend.agents.router import Router, begin_turn
from my_friend.agents.sparring_partner import SparringPartner
from my_friend.agents.validator import Validator

from .routing import route_after_reflection, route_after_sparring, route_by_mode
from .state import GraphState


def _serde():
    """상태에 담기는 pydantic 타입을 msgpack 역직렬화 allowlist에 명시 등록한 serializer.

    langgraph-checkpoint ≥ 4 는 미등록 타입 역직렬화를 경고하고 향후 차단한다(임의 클래스 로딩 방어).
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from pydantic import BaseModel

    from my_friend.agents import schemas

    allow = tuple(
        (obj.__module__, obj.__name__)
        for obj in vars(schemas).values()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    )
    return JsonPlusSerializer(allowed_msgpack_modules=allow)


def make_checkpointer(db_path: Optional[Union[str, Path]] = None) -> BaseCheckpointSaver:
    """db_path 없으면 MemorySaver(테스트/일회성), 있으면 SqliteSaver(재시작해도 세션 유지).

    동기 전용: invoke/get_state 경로(CLI, Session). astream을 쓰는 UI는 make_async_checkpointer.
    """
    if db_path is None:
        return MemorySaver(serde=_serde())
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # 동기 노드가 스레드풀에서 돌므로 커넥션을 스레드 간 공유 (SqliteSaver 내부 lock으로 직렬화)
    return SqliteSaver(sqlite3.connect(str(db_path), check_same_thread=False), serde=_serde())


async def make_async_checkpointer(db_path: Union[str, Path]) -> BaseCheckpointSaver:
    """astream/aget_state용 AsyncSqliteSaver. aiosqlite 커넥션은 실행 중인 이벤트 루프에서 열어야 한다."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver(await aiosqlite.connect(str(db_path)), serde=_serde())


def build_graph(deps: AgentDeps, checkpointer: Optional[BaseCheckpointSaver] = None):
    g = StateGraph(GraphState)
    g.add_node("turn_start", begin_turn)
    g.add_node("router", Router(deps))
    g.add_node("paper_scout", PaperScout(deps))
    g.add_node("sparring_partner", SparringPartner(deps))
    g.add_node("planner", Planner(deps))
    g.add_node("human_review", human_review)
    g.add_node("executor", Executor(deps))
    g.add_node("partial_executor", PartialExecutor(deps))
    g.add_node("validator", Validator(deps))
    g.add_node("reflector", Reflector(deps))

    # 시작은 무조건 scout: 최신 정보를 먼저 확보한 뒤 router가 그 맥락으로 경중을 판단
    g.add_edge(START, "turn_start")
    g.add_edge("turn_start", "paper_scout")
    g.add_edge("paper_scout", "router")
    g.add_conditional_edges("router", route_by_mode, ["sparring_partner", "planner"])
    g.add_conditional_edges("sparring_partner", route_after_sparring, ["human_review", "planner"])
    g.add_edge("planner", "human_review")
    # human_review → Command(goto=...) 로 분기
    g.add_edge("executor", "validator")
    g.add_edge("partial_executor", "validator")
    # 수렴점: 어느 트랙이든 코드를 짠 뒤엔 validator → reflector(최종 검수)를 거쳐 사용자에게 피드백
    g.add_edge("validator", "reflector")
    g.add_conditional_edges("reflector", route_after_reflection, ["partial_executor", "human_review", END])

    return g.compile(checkpointer=checkpointer or make_checkpointer())
