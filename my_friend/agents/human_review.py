"""사용자 승인 게이트 (LangGraph interrupt).

주의: interrupt()로 재개되면 노드가 '처음부터' 다시 실행된다 → interrupt 이전에 부수효과 금지.
여기서는 payload 구성(순수 함수)만 하고 모든 상태 변경은 Command.update로 반환한다.

resume 값: {"action": "approve" | "revise" | "reject" | "escalate", "feedback": "..."}
"""
# NOTE: `from __future__ import annotations` 미사용 — LangGraph가 Command[Literal[...]] 반환 타입을
# 런타임에 읽어 그래프 엣지(시각화/검증)를 구성한다.
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from my_friend.graph.state import GraphState

from .planner import render_spec_md
from .reflector import render_reflection
from .schemas import ReviewDecision
from .sparring_partner import render_turn

ReviewKind = Literal["proposal", "plan", "reflection"]

OPTIONS = {
    "proposal": {
        "approve": "제안된 변경을 partial_executor로 적용",
        "revise": "피드백/답변을 주고 sparring 계속",
        "escalate": "full build(planner)로 전환",
        "reject": "종료",
    },
    "plan": {"approve": "명세대로 구현(executor)", "revise": "피드백 반영해 재기획", "reject": "종료"},
    "reflection": {
        "approve": "reflector 제안 수정 적용",
        "revise": "피드백을 지시사항에 추가해 적용",
        "reject": "현 상태로 종료",
    },
}


def review_kind(state: GraphState) -> ReviewKind:
    refl = state.get("reflection")
    if refl is not None and refl.action == "ask_user":
        return "reflection"
    return "plan" if state.get("mode") == "full_build" else "proposal"


def build_payload(state: GraphState) -> dict:
    kind = review_kind(state)
    if kind == "plan":
        body = render_spec_md(state["spec"])
    elif kind == "reflection":
        body = render_reflection(state["reflection"])
    else:
        body = render_turn(state["sparring"])
    return {"kind": kind, "body": body, "options": OPTIONS[kind]}


def human_review(
    state: GraphState,
) -> Command[Literal["sparring_partner", "planner", "executor", "partial_executor", "__end__"]]:
    kind = review_kind(state)
    raw = interrupt(build_payload(state))
    decision = ReviewDecision.model_validate(raw if isinstance(raw, dict) else {"action": "revise", "feedback": str(raw)})
    if decision.action not in OPTIONS[kind]:
        decision = ReviewDecision(action="revise", feedback=decision.feedback)

    update: dict = {"review": decision}
    if decision.feedback:
        update["messages"] = [{"role": "user", "name": "user", "content": decision.feedback}]

    a = decision.action
    if a == "reject":
        return Command(goto=END, update=update)

    if kind == "proposal":
        if a == "escalate":
            return Command(goto="planner", update={**update, "mode": "full_build"})
        if a == "approve":
            # 합의된 변경이 없으면(순수 토론) 구현할 것이 없으므로 종료
            return Command(goto="partial_executor" if state.get("change") else END, update=update)
        return Command(goto="sparring_partner", update=update)

    if kind == "plan":
        return Command(goto="executor" if a == "approve" else "planner", update=update)

    # reflection
    change = state["reflection"].change
    if change is None:
        return Command(goto=END, update=update)
    if a == "revise" and decision.feedback:
        change = change.model_copy(update={"instructions": f"{change.instructions}\n\nUser direction: {decision.feedback}"})
    return Command(
        goto="partial_executor",
        update={**update, "change": change, "iteration": state.get("iteration", 0) + 1},
    )
