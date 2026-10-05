"""GraphRunner async 흐름: 실제 그래프(scripted LLM + subprocess 샌드박스)를 astream으로 구동."""
import asyncio
import sys
from collections import deque

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="interrupt() in async+sync nodes needs Python >= 3.11")
pytest.importorskip("torch")

from my_friend.agents.base import AgentDeps  # noqa: E402
from my_friend.agents.schemas import CodeBundle, CodeFile, Reflection, RouteDecision  # noqa: E402
from my_friend.graph.builder import build_graph  # noqa: E402
from my_friend.tools.sandbox import SubprocessSandbox  # noqa: E402
from tests.graph.fixtures import CAUSAL_FORWARD, ScriptedLLM, model_source, tiny_spec  # noqa: E402
from ui.runner import GraphRunner  # noqa: E402


class RecordingPresenter:
    def __init__(self, decisions=()):
        self.events = []
        self.decisions = deque(decisions)
        self.reviews = []

    async def node_started(self, task_id, node):
        self.events.append(("start", node))

    async def node_finished(self, task_id, node, result, error):
        self.events.append(("end", node, error))

    async def render_update(self, node, update, state):
        self.events.append(("update", node, sorted(update), bool(state.get("spec"))))

    async def ask_review(self, payload):
        self.reviews.append(payload["kind"])
        return self.decisions.popleft() if self.decisions else None

    async def notice(self, text):
        self.events.append(("notice", text))


def full_build_llm(**extra):
    return ScriptedLLM({
        RouteDecision: [RouteDecision(mode="full_build", reason="new")],
        type(tiny_spec()): [tiny_spec(), tiny_spec(name="v2")],
        CodeBundle: [CodeBundle(files=[CodeFile(path="model.py", content=model_source(CAUSAL_FORWARD))],
                                entry_file="model.py", notes="built")],
        Reflection: [Reflection(diagnosis="ok", risks=[], action="accept", change=None)],
        **extra,
    })


def make_runner(llm):
    graph = build_graph(AgentDeps(llm=llm, sandbox=SubprocessSandbox(timeout_s=120)))
    return GraphRunner(graph, "t1")


def test_stream_review_loop_with_buttons():
    llm = full_build_llm()
    runner = make_runner(llm)
    p = RecordingPresenter([{"action": "revise", "feedback": "v2로"}, {"action": "approve", "feedback": ""}])
    asyncio.run(runner.handle_user_message("모델 설계", p))

    assert p.reviews == ["plan", "plan"]
    started = [e[1] for e in p.events if e[0] == "start"]
    # human_review가 interrupt마다 2번 시작: interrupt 시 1회 + resume 시 노드를 '처음부터' 재실행 1회
    assert started == ["router", "planner", "human_review", "human_review", "planner", "human_review",
                       "human_review", "executor", "validator", "reflector"]
    # 모든 start에는 대응하는 end가 있어야 UI 스피너가 남지 않는다
    assert sorted(e[1] for e in p.events if e[0] == "start") == sorted(e[1] for e in p.events if e[0] == "end")
    # validator 렌더 시에는 spec이 포함된 전체 state가 전달된다
    assert ("update", "validator", ["messages", "validation"], True) in p.events
    assert llm.remaining() == {}


def test_review_timeout_then_free_text_is_revise():
    llm = full_build_llm()
    runner = make_runner(llm)
    p1 = RecordingPresenter([])                                 # 버튼 타임아웃 → None
    asyncio.run(runner.handle_user_message("모델 설계", p1))
    assert p1.events[-1][0] == "notice" and "승인 대기" in p1.events[-1][1]
    assert asyncio.run(runner.pending_review())["kind"] == "plan"

    p2 = RecordingPresenter([{"action": "approve", "feedback": ""}])
    asyncio.run(runner.handle_user_message("v2로 바꿔줘", p2))  # 자유 입력 → revise로 재개
    planner_prompt = [u for n, u, _ in llm.calls if n == "ModelSpec"][-1]
    assert "v2로 바꿔줘" in planner_prompt
    assert asyncio.run(runner.pending_review()) is None


def test_node_exception_is_reported_and_retry_resumes():
    llm = full_build_llm()
    llm.queues[CodeBundle].appendleft(RuntimeError("boom"))     # 첫 executor 호출 실패
    orig_parse = llm.parse

    def parse(system, user, output_model, **kw):
        r = orig_parse(system, user, output_model, **kw)
        if isinstance(r, Exception):
            raise r
        return r

    llm.parse = parse
    runner = make_runner(llm)
    p = RecordingPresenter([{"action": "approve", "feedback": ""}])
    asyncio.run(runner.handle_user_message("모델 설계", p))
    assert ("end", "executor", "RuntimeError") in p.events
    assert "RuntimeError: boom" in p.events[-1][1]
    assert asyncio.run(runner.failed_next()) == ("executor",)

    p2 = RecordingPresenter()
    asyncio.run(runner.handle_user_message("/retry", p2))
    assert [e[1] for e in p2.events if e[0] == "start"] == ["executor", "validator", "reflector"]
    assert asyncio.run(runner.failed_next()) == ()


def test_concurrent_message_is_rejected_while_running():
    llm = full_build_llm()
    runner = make_runner(llm)

    async def scenario():
        p1 = RecordingPresenter([{"action": "approve", "feedback": ""}])
        p2 = RecordingPresenter()
        t1 = asyncio.create_task(runner.handle_user_message("모델 설계", p1))
        await asyncio.sleep(0)                                   # t1이 lock을 잡도록 양보
        await runner.handle_user_message("끼어들기", p2)
        await t1
        return p2

    p2 = asyncio.run(scenario())
    assert p2.events == [("notice", "이전 요청을 처리 중입니다. 완료 후 다시 보내주세요.")]


def test_event_loop_not_blocked_by_sync_nodes():
    """동기 노드(샌드박스 subprocess, LLM 호출)는 스레드풀에서 실행 → 이벤트 루프가 계속 돈다."""
    llm = full_build_llm()
    runner = make_runner(llm)

    async def scenario():
        ticks = 0
        stop = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await runner.handle_user_message("모델 설계", RecordingPresenter([{"action": "approve", "feedback": ""}]))
        elapsed = loop.time() - t0
        stop.set()
        await t
        return ticks, elapsed

    ticks, elapsed = asyncio.run(scenario())
    # validator subprocess(torch import)만 해도 ~1s 이상 → 블로킹이었다면 tick이 거의 없어야 함
    assert elapsed > 0.5 and ticks > elapsed / 0.01 * 0.3
