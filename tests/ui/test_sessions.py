"""세션 목록/복원 (#11): 체크포인트만으로 스레드를 찾고, 승인 대기 세션을 이어서 실행한다."""
import asyncio

import pytest

pytest.importorskip("torch")

from my_friend.graph.builder import make_async_checkpointer, make_checkpointer  # noqa: E402
from ui.runner import GraphRunner  # noqa: E402
from ui.sessions import recent_sessions  # noqa: E402

from .test_runner import RecordingPresenter, full_build_llm, make_runner  # noqa: E402


async def _two_pending_sessions(checkpointer):
    a = make_runner(full_build_llm(), checkpointer)
    a.config["configurable"]["thread_id"] = "thread-a"
    await a.handle_user_message("첫 번째 세션: causal conv 모델 설계", RecordingPresenter([]))   # plan 승인 대기
    b = GraphRunner(make_runner(full_build_llm(), checkpointer).graph, "thread-b")   # 같은 체크포인터, 새 스레드
    await b.handle_user_message("두 번째 세션 " + "아주 긴 제목 " * 10, RecordingPresenter([]))
    return a, b


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
def test_recent_sessions_lists_threads_newest_first(tmp_path, backend):
    async def scenario():
        ckpt = await make_async_checkpointer(tmp_path / "c.sqlite") if backend == "sqlite" else make_checkpointer()
        a, b = await _two_pending_sessions(ckpt)
        got = await recent_sessions(b.graph)
        excluded = await recent_sessions(b.graph, exclude="thread-b")
        empty = await recent_sessions(b.graph, exclude="nope", limit=1)
        if backend == "sqlite":
            await ckpt.conn.close()
        return got, excluded, empty

    got, excluded, empty = asyncio.run(scenario())
    assert [s.thread_id for s in got] == ["thread-b", "thread-a"]
    assert got[1].title == "첫 번째 세션: causal conv 모델 설계"
    assert got[0].title.endswith("…") and len(got[0].title) == 40
    assert all(s.pending_review for s in got) and got[0].label.startswith("⏸ ")
    assert [s.thread_id for s in excluded] == ["thread-a"]
    assert len(empty) == 1


def test_restored_runner_resumes_pending_review_after_restart(tmp_path):
    """재시작(새 커넥션/그래프) 후 목록에서 고른 세션의 승인 대기를 다시 묻고 끝까지 실행."""
    db = tmp_path / "c.sqlite"

    async def scenario():
        ckpt = await make_async_checkpointer(db)
        await _two_pending_sessions(ckpt)
        await ckpt.conn.close()

        ckpt2 = await make_async_checkpointer(db)
        graph = make_runner(full_build_llm(), ckpt2).graph
        (picked,) = [s for s in await recent_sessions(graph) if s.thread_id == "thread-a"]
        runner = GraphRunner(graph, picked.thread_id)
        state = await runner.state()
        p = RecordingPresenter([{"action": "approve", "feedback": ""}])
        await runner.resume_pending(p)
        after = await runner.state()
        await ckpt2.conn.close()
        return state, p, after

    state, p, after = asyncio.run(scenario())
    assert state["messages"][0]["content"] == "첫 번째 세션: causal conv 모델 설계" and state["spec"] is not None
    assert p.reviews == ["plan"]
    assert [e[1] for e in p.events if e[0] == "start"][-3:] == ["executor", "validator", "reflector"]
    assert after["validation"].status == "pass"


def test_resume_pending_is_noop_without_pending_review(tmp_path):
    async def scenario():
        runner = make_runner(full_build_llm())
        p = RecordingPresenter()
        await runner.resume_pending(p)
        return p

    p = asyncio.run(scenario())
    assert p.events == [] and p.reviews == []
