"""LangGraph async 스트리밍 ↔ UI 브리지 (Chainlit 비의존).

Presenter 프로토콜만 구현하면 Chainlit / 테스트 / 다른 UI에서 동일한 루프를 쓴다.

한 번의 사용자 입력 처리 = run():
    astream(payload) ─ tasks 이벤트 → presenter.node_started / node_finished  (진행 표시)
                     ─ updates 이벤트 → presenter.render_update                 (노드 결과)
                     ─ __interrupt__  → presenter.ask_review → Command(resume) 로 재개, 반복
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Protocol

from langgraph.types import Command

RECURSION_LIMIT = 80   # reflector→partial→validator 루프가 기본 25 superstep을 쉽게 넘는다


class Presenter(Protocol):
    async def node_started(self, task_id: str, node: str) -> None: ...
    async def node_finished(self, task_id: str, node: str, result: Any, error: Optional[str]) -> None: ...
    async def render_update(self, node: str, update: Dict[str, Any], state: Dict[str, Any]) -> None: ...
    async def ask_review(self, payload: Dict[str, Any]) -> Optional[Dict[str, str]]: ...
    async def notice(self, text: str) -> None: ...


class GraphRunner:
    def __init__(self, graph, thread_id: str) -> None:
        self.graph = graph
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        # 같은 thread_id에 대한 동시 실행은 체크포인트 순서를 꼬이게 한다 → 세션당 직렬화
        self.lock = asyncio.Lock()

    async def pending_review(self) -> Optional[dict]:
        snap = await self.graph.aget_state(self.config)    # get_state(동기)는 이벤트 루프를 막는다
        for task in snap.tasks if snap else ():
            if task.interrupts:
                return task.interrupts[0].value
        return None

    async def failed_next(self) -> tuple:
        """예외로 중단된 run이 남긴 '다음 실행 노드'. interrupt 대기와 구분된다."""
        snap = await self.graph.aget_state(self.config)
        if not snap or not snap.next or any(t.interrupts for t in snap.tasks):
            return ()
        return tuple(snap.next)

    async def state(self) -> Dict[str, Any]:
        snap = await self.graph.aget_state(self.config)
        return dict(snap.values) if snap and snap.values else {}

    async def handle_user_message(
        self, text: str, presenter: Presenter, mode: Optional[str] = None, max_iterations: Optional[int] = None
    ) -> None:
        if self.lock.locked():
            await presenter.notice("이전 요청을 처리 중입니다. 완료 후 다시 보내주세요.")
            return
        async with self.lock:
            if await self.pending_review() is not None:
                # 버튼 대기가 타임아웃된 뒤 들어온 자유 입력 = 수정 요청으로 해석
                payload: Any = Command(resume={"action": "revise", "feedback": text})
            elif text.strip() == "/retry" and await self.failed_next():
                payload = None          # None 입력 = 마지막 체크포인트에서 실패한 노드부터 재실행
            else:
                payload = {"request": text, "requested_mode": mode}
                if max_iterations is not None:
                    payload["max_iterations"] = max_iterations
            await self._drive(payload, presenter)

    async def resume_pending(self, presenter: Presenter) -> None:
        """복원된 세션이 승인 대기 중이면 같은 리뷰를 다시 묻고 이어서 실행 (#11)."""
        if self.lock.locked():
            await presenter.notice("이전 요청을 처리 중입니다. 완료 후 다시 시도해주세요.")
            return
        async with self.lock:
            pending = await self.pending_review()
            if pending is None:
                return
            decision = await presenter.ask_review(pending)
            if decision is None:
                await presenter.notice("⏸ 승인 대기 중입니다. 메시지를 입력하면 '수정 요청(revise)'으로 전달됩니다.")
                return
            await self._drive(Command(resume=decision), presenter)

    async def _drive(self, payload: Any, presenter: Presenter) -> None:
        while True:
            interrupt_value = await self._stream(payload, presenter)
            if interrupt_value is None:
                return
            decision = await presenter.ask_review(interrupt_value)
            if decision is None:
                await presenter.notice("⏸ 승인 대기 중입니다. 메시지를 입력하면 '수정 요청(revise)'으로 전달됩니다.")
                return
            payload = Command(resume=decision)

    async def _stream(self, payload: Any, presenter: Presenter) -> Optional[dict]:
        interrupt_value = None
        open_tasks: Dict[str, str] = {}
        try:
            async for mode, chunk in self.graph.astream(payload, self.config, stream_mode=["tasks", "updates"]):
                if mode == "tasks":
                    if "triggers" in chunk:                                   # task 시작
                        open_tasks[chunk["id"]] = chunk["name"]
                        await presenter.node_started(chunk["id"], chunk["name"])
                    else:                                                     # task 종료
                        open_tasks.pop(chunk["id"], None)
                        await presenter.node_finished(chunk["id"], chunk["name"], chunk.get("result"), chunk.get("error"))
                    continue
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        interrupt_value = update[0].value
                    elif update:
                        # 'updates'는 노드가 반환한 delta만 담는다. 렌더러가 전체 맥락(spec 등)이 필요할 때만 조회.
                        state = await self.state() if node == "validator" else {}
                        await presenter.render_update(node, update, state)
        except asyncio.CancelledError:
            # Stop 버튼: astream 대기는 취소되지만, 스레드풀에서 돌던 동기 노드(LLM 호출/샌드박스)는
            # 끝까지 실행된다. 해당 superstep은 커밋되지 않으므로 상태는 직전 체크포인트에 머문다.
            for tid, name in open_tasks.items():
                await asyncio.shield(presenter.node_finished(tid, name, None, "cancelled"))
            raise
        except Exception as e:  # noqa: BLE001 — 노드 예외를 UI에 노출하고 세션은 유지
            for tid, name in open_tasks.items():
                await presenter.node_finished(tid, name, None, f"{type(e).__name__}")
            await presenter.notice(f"❗ `{type(e).__name__}: {e}`\n\n상태는 마지막 체크포인트로 유지됩니다. `/retry`로 실패한 노드부터 재시도할 수 있습니다.")
            return None
        return interrupt_value
