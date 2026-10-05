from __future__ import annotations

from my_friend.graph.state import PER_TURN_RESET, GraphState

from . import prompts
from .base import AgentDeps, compose, render_history, section
from .schemas import RouteDecision

DEFAULT_MAX_ITERATIONS = 3


class Router:
    """턴 진입점: 턴 단위 상태 초기화 + 모드 결정 (사용자 지정 > LLM 분류)."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: GraphState) -> dict:
        request = state["request"]
        update: dict = {
            **PER_TURN_RESET,
            "requested_mode": None,     # 1회성 오버라이드 소비
            "max_iterations": state.get("max_iterations") or DEFAULT_MAX_ITERATIONS,
            "messages": [{"role": "user", "name": "user", "content": request}],
        }
        if state.get("requested_mode"):
            update.update(mode=state["requested_mode"], route_reason="user override")
            return update

        files = state.get("files") or {}
        workspace = f"files={sorted(files)}; spec={'yes: ' + state['spec'].name if state.get('spec') else 'none'}"
        decision = self.deps.llm.parse(
            prompts.ROUTER,
            compose(
                section("conversation", render_history(state, 6)),
                section("workspace", workspace),
                section("request", request),
            ),
            RouteDecision,
            effort=self.deps.effort["router"],
            max_tokens=4000,
        )
        update.update(mode=decision.mode, route_reason=decision.reason)
        return update
