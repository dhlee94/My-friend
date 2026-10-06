from __future__ import annotations

from my_friend.graph.state import GraphState

from . import prompts
from .base import AgentDeps, assistant_turn, compose, render_files, render_history, render_scout, section
from .schemas import SparringTurn


def render_turn(t: SparringTurn) -> str:
    out = [t.reply]
    for i, o in enumerate(t.options, 1):
        out.append(f"\n**옵션 {i}. {o.option}** — 복잡도: {o.complexity}\n  + {o.pros}\n  − {o.cons}")
    if t.recommendation:
        out.append(f"\n**추천:** {t.recommendation}")
    if t.questions:
        out.append("\n" + "\n".join(f"Q. {q}" for q in t.questions))
    if t.escalate:
        out.append("\n**→ 새 아키텍처가 필요해 Full-Build(Planner)로 전환합니다.**")
    elif t.proposed_change:
        c = t.proposed_change
        out.append(f"\n**제안 변경:** {c.summary}\n  대상: {', '.join(c.target_symbols)}\n  shape: {'; '.join(c.expected_shape_changes)}")
    return "\n".join(out)


class SparringPartner:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: GraphState) -> dict:
        turn = self.deps.llm.parse(
            prompts.SPARRING,
            compose(
                section("scout_digest", render_scout(state)),
                section("code", render_files(state.get("files") or {}, self.deps.max_file_chars)),
                section("conversation", render_history(state, self.deps.history_turns)),
                section("request", state["request"]),
            ),
            SparringTurn,
            effort=self.deps.effort["sparring_partner"],
            max_tokens=16000,
        )
        update: dict = {
            "sparring": turn,
            "change": None if turn.escalate else turn.proposed_change,
            "messages": [assistant_turn("sparring_partner", render_turn(turn))],
        }
        if turn.escalate:
            update.update(mode="full_build", route_reason="escalated by sparring_partner")
        return update
