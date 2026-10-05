"""그래프 조립.

    START → router ─┬─ sparring ──→ paper_scout → sparring_partner → human_review ⟲(revise)
                    │                                                  ├ approve  → partial_executor → validator ─┬ pass → END
                    │                                                  └ escalate → planner                        └ fail → reflector
                    └─ full_build → planner → human_review ⟲(revise)
                                                 └ approve → executor → validator → reflector ─┬ patch    → partial_executor → validator …
                                                                                               ├ ask_user → human_review
                                                                                               └ accept / budget 초과 → END
"""
from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from my_friend.agents.base import AgentDeps
from my_friend.agents.executor import Executor, PartialExecutor
from my_friend.agents.human_review import human_review
from my_friend.agents.paper_scout import PaperScout
from my_friend.agents.planner import Planner
from my_friend.agents.reflector import Reflector
from my_friend.agents.router import Router
from my_friend.agents.sparring_partner import SparringPartner
from my_friend.agents.validator import Validator

from .routing import route_after_reflection, route_after_validation, route_by_mode
from .state import GraphState


def make_checkpointer() -> BaseCheckpointSaver:
    """상태에 담기는 pydantic 타입을 msgpack 역직렬화 allowlist에 명시 등록한 MemorySaver.

    langgraph-checkpoint ≥ 4 는 미등록 타입 역직렬화를 경고하고 향후 차단한다(임의 클래스 로딩 방어).
    구버전(≤ 2.x, Python 3.9 환경)은 해당 인자가 없으므로 기본 serializer를 쓴다.
    """
    import inspect

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from pydantic import BaseModel

    from my_friend.agents import schemas

    if "allowed_msgpack_modules" not in inspect.signature(JsonPlusSerializer.__init__).parameters:
        return MemorySaver()
    allow = tuple(
        (obj.__module__, obj.__name__)
        for obj in vars(schemas).values()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    )
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allow))


def build_graph(deps: AgentDeps, checkpointer: Optional[BaseCheckpointSaver] = None):
    g = StateGraph(GraphState)
    g.add_node("router", Router(deps))
    g.add_node("paper_scout", PaperScout(deps))
    g.add_node("sparring_partner", SparringPartner(deps))
    g.add_node("planner", Planner(deps))
    g.add_node("human_review", human_review)
    g.add_node("executor", Executor(deps))
    g.add_node("partial_executor", PartialExecutor(deps))
    g.add_node("validator", Validator(deps))
    g.add_node("reflector", Reflector(deps))

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_by_mode, ["paper_scout", "planner"])
    g.add_edge("paper_scout", "sparring_partner")
    g.add_edge("sparring_partner", "human_review")
    g.add_edge("planner", "human_review")
    # human_review → Command(goto=...) 로 분기
    g.add_edge("executor", "validator")
    g.add_edge("partial_executor", "validator")
    g.add_conditional_edges("validator", route_after_validation, ["reflector", END])
    g.add_conditional_edges("reflector", route_after_reflection, ["partial_executor", "human_review", END])

    # MemorySaver: 프로세스 메모리 — 재시작 시 소실. 영속화는 SqliteSaver로 교체 (동일 인터페이스).
    return g.compile(checkpointer=checkpointer or make_checkpointer())
