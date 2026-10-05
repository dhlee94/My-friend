from __future__ import annotations

import json

from my_friend.graph.state import GraphState

from . import prompts
from .base import AgentDeps, assistant_turn, compose, render_history, render_spec, section
from .schemas import ModelSpec


def render_spec_md(s: ModelSpec) -> str:
    dims = ", ".join(f"{d.symbol}={d.value} ({d.meaning})" for d in s.dims)
    mods = "\n".join(
        f"| `{m.path}` | {m.role} | {'×'.join(m.input_shape)} | {'×'.join(m.output_shape)} | {m.complexity} |"
        for m in s.modules
    )
    return "\n".join([
        f"### {s.name}\n{s.summary}",
        f"**dims:** {dims}",
        f"**inputs:** " + ", ".join(f"{t.name}: {'×'.join(t.shape)} {t.dtype}" for t in s.inputs),
        f"**outputs:** " + ", ".join(f"{t.name}: {'×'.join(t.shape)}" for t in s.outputs),
        f"**causal axis:** {s.causal_axis or '-'} · **train batch:** {s.train_batch_size}",
        "| module | role | in | out | complexity |\n|---|---|---|---|---|\n" + mods,
        f"```\n{s.shape_diagram}\n```",
        "**설계 결정:**\n" + "\n".join(f"- {d}" for d in s.design_decisions),
        "**미결 질문:**\n" + "\n".join(f"- {q}" for q in s.open_questions),
    ])


class Planner:
    def __init__(self, deps: AgentDeps, rag_top_k: int = 4) -> None:
        self.deps = deps
        self.rag_top_k = rag_top_k

    def __call__(self, state: GraphState) -> dict:
        refs = ""
        if self.deps.retriever is not None:
            refs = self.deps.retriever.retrieve(state["request"], top_k=self.rag_top_k).to_context()
        spec = self.deps.llm.parse(
            prompts.PLANNER,
            compose(
                section("reference_passages", refs),
                # revise 루프: 직전 명세 + (history에 추가된) 사용자 피드백을 반영해 수정
                section("previous_spec", render_spec(state)),
                section("conversation", render_history(state, self.deps.history_turns)),
                section("request", state["request"]),
            ),
            ModelSpec,
            effort=self.deps.effort["planner"],
            max_tokens=32000,
        )
        try:
            json.loads(spec.init_kwargs_json or "{}")
        except json.JSONDecodeError as e:
            # 승인 단계에서 사용자에게 드러나도록 미결 질문으로 올린다 (validator까지 가서 터지지 않게)
            spec.open_questions.append(f"init_kwargs_json 파싱 실패({e.msg}) — 수정 필요")
        return {"spec": spec, "messages": [assistant_turn("planner", render_spec_md(spec))]}
