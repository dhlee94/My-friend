"""조건부 엣지 함수 (순수 함수 — 단위 테스트 대상)."""
from __future__ import annotations

from langgraph.graph import END

from .state import GraphState


def route_by_mode(state: GraphState) -> str:
    """scout 이후 분기: 가벼우면 sparring_partner, 무거우면 planner."""
    return "sparring_partner" if state["mode"] == "sparring" else "planner"


def route_after_sparring(state: GraphState) -> str:
    """에스컬레이션 엣지: sparring 중 새 아키텍처가 필요하다고 판단되면 승인 없이 planner로 직행."""
    turn = state.get("sparring")
    return "planner" if turn is not None and turn.escalate else "human_review"


def route_after_reflection(state: GraphState) -> str:
    r = state["reflection"]
    if r.action == "ask_user":
        return "human_review"
    if r.action == "patch":
        # reflector가 patch 시 iteration을 증가시킴 → 예산 초과면 사람에게 넘기지 않고 종료(리포트 남김)
        return "partial_executor" if state.get("iteration", 0) <= state.get("max_iterations", 3) else END
    return END
