from __future__ import annotations

from my_friend.graph.state import GraphState

from . import prompts
from .base import AgentDeps, assistant_turn, compose, render_files, render_history, render_spec, render_validation, section
from .schemas import Reflection


def render_reflection(r: Reflection) -> str:
    out = [r.diagnosis]
    out += [f"- [{x.severity}/{x.category}] {x.evidence} → {x.fix}" for x in r.risks]
    out.append(f"action: {r.action}" + (f" — {r.change.summary}" if r.change else ""))
    return "\n".join(out)


class Reflector:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: GraphState) -> dict:
        it, max_it = state.get("iteration", 0), state.get("max_iterations", 3)
        refl = self.deps.llm.parse(
            prompts.REFLECTOR,
            compose(
                section("validation_report", render_validation(state)),
                section("spec", render_spec(state)),
                section("code", render_files(state.get("files") or {}, self.deps.max_file_chars)),
                section("conversation", render_history(state, 6)),
                section("budget", f"fix iteration {it}/{max_it}"),
            ),
            Reflection,
            effort=self.deps.effort["reflector"],
            max_tokens=32000,
        )
        if refl.action == "patch" and refl.change is None:
            refl.action = "ask_user"            # 고칠 내용 없는 patch는 사용자 판단으로
        update: dict = {"reflection": refl, "messages": [assistant_turn("reflector", render_reflection(refl))]}
        if refl.action == "patch":
            update["change"] = refl.change
            update["iteration"] = it + 1
        return update
