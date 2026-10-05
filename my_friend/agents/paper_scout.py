from __future__ import annotations

import logging

from my_friend.graph.state import GraphState

from . import prompts
from .base import AgentDeps, assistant_turn, compose, render_history, section
from .schemas import ScoutReport, ScoutSearchPlan

log = logging.getLogger(__name__)


class PaperScout:
    """arXiv 최신 논문 + 로컬 RAG(논문/코드) 탐색 → 트렌드 요약.

    외부 검색 실패는 그래프를 멈추지 않는다: 빈 결과로 진행하고 digest에 사유를 남긴다.
    """

    def __init__(self, deps: AgentDeps, max_papers: int = 10, local_top_k: int = 4) -> None:
        self.deps = deps
        self.max_papers = max_papers
        self.local_top_k = local_top_k

    def __call__(self, state: GraphState) -> dict:
        request = state["request"]
        effort = self.deps.effort["paper_scout"]
        plan = self.deps.llm.parse(
            prompts.SCOUT_PLAN,
            compose(section("conversation", render_history(state, 4)), section("request", request)),
            ScoutSearchPlan,
            effort=effort,
            max_tokens=4000,
        )

        notes = []
        entries = []
        if self.deps.arxiv is not None:
            try:
                entries = self.deps.arxiv.search(plan.arxiv_query, self.max_papers)
            except Exception as e:  # noqa: BLE001 — 네트워크/파싱 실패는 degrade
                log.warning("arxiv search failed: %s", e)
                notes.append(f"arXiv 검색 실패: {type(e).__name__}")
        local = ""
        if self.deps.retriever is not None:
            local = self.deps.retriever.retrieve(plan.local_query, top_k=self.local_top_k).to_context()

        if not entries and not local:
            report = ScoutReport(papers=[], trends=[], digest="관련 논문을 찾지 못했습니다. " + " ".join(notes))
        else:
            report = self.deps.llm.parse(
                prompts.SCOUT_SUMMARIZE,
                compose(
                    section("request", request),
                    section("arxiv_query", plan.arxiv_query),
                    section("arxiv_results", "\n\n".join(e.render() for e in entries)),
                    section("local_passages", local),
                ),
                ScoutReport,
                effort=effort,
                max_tokens=16000,
            )
            # 환각 방어: 검색 결과에 없는 arXiv id는 제거
            known = {e.arxiv_id.split("v")[0] for e in entries}
            report.papers = [p for p in report.papers if p.arxiv_id.split("v")[0] in known]

        lines = [report.digest] + [f"- [{p.arxiv_id}] {p.title}: {p.key_idea}" for p in report.papers]
        return {"scout": report, "messages": [assistant_turn("paper_scout", "\n".join(lines))]}
