"""노드 공용 의존성 & 컨텍스트 렌더링."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from my_friend.core.llm import StructuredLLM
from my_friend.tools.arxiv import ArxivSearcher
from my_friend.tools.code_patch import outline
from my_friend.tools.sandbox import Sandbox, SubprocessSandbox

if TYPE_CHECKING:
    from my_friend.graph.state import GraphState
    from my_friend.rag.retriever import HybridRetriever

# 노드별 effort: 라우팅/검색 계획은 가볍게, 설계·구현·디버깅은 깊게.
DEFAULT_EFFORT: Dict[str, str] = {
    "router": "low",
    "paper_scout": "low",
    "sparring_partner": "medium",
    "planner": "high",
    "executor": "high",
    "partial_executor": "high",
    "reflector": "high",
}


@dataclass
class AgentDeps:
    llm: StructuredLLM                    # 기본 LLM (llms에 없는 노드가 사용)
    sandbox: Sandbox = field(default_factory=SubprocessSandbox)
    arxiv: Optional[ArxivSearcher] = None
    retriever: Optional["HybridRetriever"] = None
    effort: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EFFORT))
    history_turns: int = 12               # 프롬프트에 넣을 최근 대화 턴 수
    max_file_chars: int = 40_000          # 파일 하나를 통째로 넣을 상한
    llms: Dict[str, StructuredLLM] = field(default_factory=dict)   # 노드별 LLM (모델이 다를 때)

    def llm_for(self, node: str) -> StructuredLLM:
        return self.llms.get(node, self.llm)


def section(tag: str, body: str) -> str:
    return f"<{tag}>\n{body.strip()}\n</{tag}>" if body and body.strip() else ""


def render_history(state: "GraphState", n: int) -> str:
    turns = state.get("messages", [])[-n:]
    return "\n".join(f"[{t['name']}] {t['content']}" for t in turns)


def render_files(files: Dict[str, str], max_chars: int, outline_only: bool = False) -> str:
    parts: List[str] = []
    for path in sorted(files):
        src = files[path]
        if outline_only or len(src) > max_chars:
            try:
                body = "\n".join(outline(src)) if path.endswith(".py") else src[:2000]
            except SyntaxError:
                body = src[:max_chars]
            parts.append(f'<file path="{path}" view="outline">\n{body}\n</file>')
        else:
            parts.append(f'<file path="{path}">\n{src}\n</file>')
    return "\n".join(parts)


def render_spec(state: "GraphState") -> str:
    spec = state.get("spec")
    return json.dumps(spec.model_dump(), ensure_ascii=False, indent=1) if spec else ""


def render_scout(state: "GraphState") -> str:
    scout = state.get("scout")
    if not scout:
        return ""
    return "\n".join(
        [scout.digest]
        + [f"- trend: {t}" for t in scout.trends]
        + [f"- [{p.arxiv_id}] {p.title} ({p.published[:10]}): {p.key_idea}" for p in scout.papers]
    )


def render_validation(state: "GraphState", max_shape_rows: int = 80) -> str:
    v = state.get("validation")
    if v is None:
        return ""
    d = v.model_dump()
    d["shape_log"] = d["shape_log"][:max_shape_rows]
    return json.dumps(d, ensure_ascii=False, indent=1)


def compose(*sections: str) -> str:
    return "\n\n".join(s for s in sections if s)


def assistant_turn(name: str, content: str) -> dict:
    return {"role": "assistant", "name": name, "content": content}
