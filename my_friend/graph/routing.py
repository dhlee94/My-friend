"""조건부 엣지 함수 (순수 함수 — 단위 테스트 대상)."""
from __future__ import annotations

from langgraph.graph import END

from .state import GraphState


def route_by_mode(state: GraphState) -> str:
    """Sparring: scout → sparring → review → partial → validator
    Full-Build: planner → review → executor → validator → reflector"""
    return "paper_scout" if state["mode"] == "sparring" else "planner"


def route_after_validation(state: GraphState) -> str:
    v = state["validation"]
    if state["mode"] == "full_build":
        # 통과해도 reflector를 거친다: 누수/OOM/수치 불안정은 forward 성공과 무관
        return "reflector"
    # sparring: 통과(또는 spec 없어 skipped)면 종료, 실패만 reflector로
    return "reflector" if v is not None and v.status == "fail" else END


def route_after_reflection(state: GraphState) -> str:
    r = state["reflection"]
    if r.action == "ask_user":
        return "human_review"
    if r.action == "patch":
        # reflector가 patch 시 iteration을 증가시킴 → 예산 초과면 사람에게 넘기지 않고 종료(리포트 남김)
        return "partial_executor" if state.get("iteration", 0) <= state.get("max_iterations", 3) else END
    return END
