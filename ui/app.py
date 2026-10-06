"""my-friend Chainlit UI.

    cd ui && chainlit run app.py --host 0.0.0.0 --port 8000

Python ≥ 3.11 필수: 3.10 이하에서는 async 실행 중 스레드풀로 넘어간 동기 노드에 LangGraph
config contextvar가 전파되지 않아 interrupt()가 "Called get_config outside of a runnable context"로 실패한다.
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

if sys.version_info < (3, 11):
    raise RuntimeError("my-friend UI requires Python >= 3.11 (LangGraph interrupt() in async + sync nodes)")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio  # noqa: E402

import chainlit as cl  # noqa: E402
from chainlit.input_widget import Select, Slider  # noqa: E402

from my_friend.agents.base import AgentDeps  # noqa: E402
from my_friend.agents.planner import render_spec_md  # noqa: E402
from my_friend.core.llm import DEFAULT_MODEL, AnthropicLLM  # noqa: E402
from my_friend.graph.builder import build_graph, make_async_checkpointer  # noqa: E402
from my_friend.tools.arxiv import ArxivClient  # noqa: E402
from my_friend.tools.sandbox import SubprocessSandbox  # noqa: E402
from ui.render import chain_conflicts, shape_table, spec_to_mermaid, validation_summary_md  # noqa: E402
from ui.runner import GraphRunner  # noqa: E402

REVIEW_TIMEOUT_S = int(os.environ.get("REVIEW_TIMEOUT_S", "3600"))
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", str(ROOT / "data" / "checkpoints.sqlite"))

NODE_LABEL = {
    "router": "🧭 Router",
    "paper_scout": "📚 Paper Scout",
    "sparring_partner": "🥊 Sparring Partner",
    "planner": "📐 Planner",
    "executor": "🛠 Executor",
    "partial_executor": "🩹 Partial Executor",
    "validator": "🧪 Validator",
    "reflector": "🔍 Reflector",
}
ACTION_LABEL = {"approve": "✅ 승인", "revise": "✏️ 수정 요청", "escalate": "📐 Full build로 전환", "reject": "⛔ 종료"}
REVIEW_TITLE = {"proposal": "변경 제안을 적용할까요?", "plan": "이 아키텍처 명세로 구현할까요?", "reflection": "Reflector 수정안을 적용할까요?"}


# --------------------------------------------------------------------------- graph
_graph = None
_graph_lock = asyncio.Lock()


def _build_graph(checkpointer):
    retriever = None
    index_dir = os.environ.get("RAG_INDEX_DIR")
    if index_dir and (Path(index_dir) / "manifest.json").exists():
        # bge-m3 로드는 수 초~수십 초 블로킹 → 호출부에서 make_async로 스레드에 넘긴다
        from my_friend.rag.embedder import SentenceTransformerEmbedder
        from my_friend.rag.index import HybridIndex
        from my_friend.rag.query_rewriter import QueryRewriter
        from my_friend.rag.retriever import HybridRetriever

        index = HybridIndex.load(index_dir)
        retriever = HybridRetriever(
            index, SentenceTransformerEmbedder(index.embedder_name), QueryRewriter(AnthropicLLM(effort="low"))
        )
    deps = AgentDeps(
        llm=AnthropicLLM(model=os.environ.get("MY_FRIEND_MODEL", DEFAULT_MODEL)),
        sandbox=SubprocessSandbox(
            device=os.environ.get("SANDBOX_DEVICE", "cpu"),
            timeout_s=float(os.environ.get("SANDBOX_TIMEOUT_S", "120")),
        ),
        arxiv=None if os.environ.get("DISABLE_ARXIV") == "1" else ArxivClient(),
        retriever=retriever,
    )
    # 그래프/체크포인터는 프로세스 전역 1개: 세션 격리는 thread_id로.
    return build_graph(deps, checkpointer)


async def get_graph():
    global _graph
    async with _graph_lock:
        if _graph is None:
            # aiosqlite 커넥션은 이벤트 루프에서 열고, 무거운 그래프 구성(bge-m3 로드)만 스레드로 넘긴다
            checkpointer = await make_async_checkpointer(CHECKPOINT_DB)
            _graph = await cl.make_async(_build_graph)(checkpointer)
    return _graph


# ----------------------------------------------------------------------- presenter
class ChainlitPresenter:
    def __init__(self) -> None:
        self.steps: Dict[str, cl.Step] = {}

    async def node_started(self, task_id: str, node: str) -> None:
        if node not in NODE_LABEL:          # human_review는 버튼 UI로 표현
            return
        step = cl.Step(name=NODE_LABEL[node], type="tool", show_input=False)
        await step.send()
        self.steps[task_id] = step

    async def node_finished(self, task_id: str, node: str, result: Any, error: Optional[str]) -> None:
        step = self.steps.pop(task_id, None)
        if step is None:
            return
        if error:
            step.output = f"❗ {error}"
            step.is_error = True
        elif node == "router" and isinstance(result, dict):
            step.output = f"mode=`{result.get('mode')}` · {result.get('route_reason', '')}"
        elif node == "validator" and isinstance(result, dict) and result.get("validation"):
            v = result["validation"]
            step.output = f"{v.status}" + (f" @ {v.failed_stage}" if v.failed_stage else "")
        else:
            step.output = "완료"
        await step.update()

    async def render_update(self, node: str, update: Dict[str, Any], state: Dict[str, Any]) -> None:
        author = NODE_LABEL.get(node, node)
        last = (update.get("messages") or [{}])[-1].get("content", "")

        if node == "paper_scout":
            scout = update["scout"]
            papers = "\n".join(
                f"- [{p.title}](https://arxiv.org/abs/{p.arxiv_id}) · {p.published[:10]} — {p.key_idea}" for p in scout.papers
            )
            trends = "\n".join(f"- {t}" for t in scout.trends)
            await cl.Message(author=author, content=f"{scout.digest}\n\n**논문**\n{papers or '-'}\n\n**트렌드**\n{trends or '-'}").send()

        elif node == "sparring_partner":
            await cl.Message(author=author, content=last).send()

        elif node == "planner":
            spec = update["spec"]
            conflicts = chain_conflicts(spec)
            warn = ("\n\n> ⚠️ **정적 shape 체인 충돌**\n" + "\n".join(f"> - {c}" for c in conflicts)) if conflicts else ""
            diagram = cl.CustomElement(
                name="Mermaid", display="inline",
                props={"code": spec_to_mermaid(spec), "title": f"{spec.name} — forward shape flow"},
            )
            await cl.Message(author=author, content=render_spec_md(spec) + warn, elements=[diagram]).send()

        elif node in ("executor", "partial_executor"):
            files = {p: s for p, s in (update.get("files") or {}).items() if s}
            elements = [cl.Text(name=p, content=s, language="python" if p.endswith(".py") else None, display="side")
                        for p, s in files.items()]
            # side 엘리먼트는 content에 이름이 등장해야 클릭 가능한 링크가 된다
            footer = f"\n\n📄 {' '.join(files)}" if files else ""
            await cl.Message(author=author, content=last + footer, elements=elements).send()

        elif node == "validator":
            v = update["validation"]
            elements = []
            if v.traceback:
                elements.append(cl.Text(name="traceback", content=v.traceback, language="python", display="side"))
            if v.stderr_tail.strip():
                elements.append(cl.Text(name="stderr", content=v.stderr_tail, display="side"))
            spec = state.get("spec")
            if spec is not None and v.shape_log:
                # 실측 결과로 노드를 색칠한 다이어그램 (명세 vs 실제)
                elements.append(cl.CustomElement(
                    name="Mermaid", display="inline",
                    props={"code": spec_to_mermaid(spec, v), "title": "실측 shape 검증 (green=일치, red=불일치)"},
                ))
            refs = " ".join(e.name for e in elements if isinstance(e, cl.Text))
            body = validation_summary_md(v) + "\n\n#### 레이어별 Shape 추적\n" + shape_table(v)
            await cl.Message(author=author, content=body + (f"\n\n📎 {refs}" if refs else ""), elements=elements).send()

        elif node == "reflector":
            r = update["reflection"]
            rows = "\n".join(
                f"| {x.severity} | {x.category} | {x.evidence.replace('|', '/')} | {x.fix.replace('|', '/')} |" for x in r.risks
            )
            table = f"\n\n| severity | category | evidence | fix |\n|---|---|---|---|\n{rows}" if r.risks else ""
            nxt = {"patch": "→ 자동 수정 진행", "accept": "→ 완료", "ask_user": "→ 사용자 판단 필요"}[r.action]
            await cl.Message(author=author, content=f"{r.diagnosis}{table}\n\n**{r.action}** {nxt}").send()

    async def ask_review(self, payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
        kind, options = payload["kind"], payload["options"]
        actions = [
            cl.Action(name="review", payload={"action": a}, label=ACTION_LABEL.get(a, a), tooltip=desc)
            for a, desc in options.items()
        ]
        guide = "\n".join(f"- **{ACTION_LABEL.get(a, a)}** — {d}" for a, d in options.items())
        res = await cl.AskActionMessage(
            content=f"### {REVIEW_TITLE.get(kind, '승인 요청')}\n{guide}", actions=actions, timeout=REVIEW_TIMEOUT_S
        ).send()
        if res is None:
            return None
        action = res["payload"]["action"]
        feedback = ""
        if action in ("revise", "escalate"):
            ans = await cl.AskUserMessage(
                content="피드백/답변을 입력하세요." + (" (비워두려면 '-')" if action == "escalate" else ""),
                timeout=REVIEW_TIMEOUT_S,
            ).send()
            if ans is None and action == "revise":
                return None
            feedback = "" if ans is None or ans["output"].strip() == "-" else ans["output"]
        return {"action": action, "feedback": feedback}

    async def notice(self, text: str) -> None:
        await cl.Message(author="my-friend", content=text).send()


# ------------------------------------------------------------------------ handlers
@cl.on_chat_start
async def on_chat_start() -> None:
    graph = await get_graph()
    # Chainlit thread id와 일치시켜 둔다: data layer를 붙이면 on_chat_resume에서 같은 체크포인트를 이어받는다
    runner = GraphRunner(graph, thread_id=cl.context.session.thread_id or uuid.uuid4().hex)
    cl.user_session.set("runner", runner)
    cl.user_session.set("settings", {"mode": "auto", "max_iterations": 3})
    await cl.ChatSettings([
        Select(id="mode", label="모드", values=["auto", "sparring", "full_build"], initial_index=0,
               description="auto: router가 질문 성격으로 결정"),
        Slider(id="max_iterations", label="Reflector 자동 수정 최대 횟수", initial=3, min=0, max=6, step=1),
    ]).send()
    workflow = cl.CustomElement(name="Mermaid", display="inline",
                                # draw_mermaid()는 <p> 라벨을 쓰는데 strict 모드에선 태그가 글자로 보이므로 제거
                                props={"code": re.sub(r"</?p>", "", graph.get_graph().draw_mermaid()),
                                       "title": "my-friend workflow"})
    await cl.Message(
        author="my-friend",
        content="논문 아이디어나 구현하고 싶은 아키텍처를 말씀해 주세요. 가벼운 질문/국소 수정은 **Sparring**, "
                "새 모델 설계는 **Full-Build** 모드로 진행됩니다. 실패한 실행은 `/retry`로 재시도합니다.",
        elements=[workflow],
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]) -> None:
    cl.user_session.set("settings", {"mode": settings.get("mode", "auto"),
                                     "max_iterations": int(settings.get("max_iterations", 3))})


@cl.on_message
async def on_message(message: cl.Message) -> None:
    runner: GraphRunner = cl.user_session.get("runner")
    settings = cl.user_session.get("settings") or {}
    mode = settings.get("mode", "auto")
    await runner.handle_user_message(
        message.content,
        ChainlitPresenter(),
        mode=None if mode == "auto" else mode,
        max_iterations=settings.get("max_iterations"),
    )
