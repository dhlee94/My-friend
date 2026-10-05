"""UI/CLI용 얇은 세션 래퍼: thread_id 고정 + interrupt/resume 왕복."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langgraph.types import Command

from my_friend.agents.schemas import Mode


@dataclass
class StepResult:
    state: Dict[str, Any]
    pending_review: Optional[dict]          # interrupt payload (kind/body/options), 없으면 None
    new_messages: List[dict]

    @property
    def done(self) -> bool:
        return self.pending_review is None


class Session:
    def __init__(self, graph, thread_id: Optional[str] = None) -> None:
        self.graph = graph
        self.thread_id = thread_id or uuid.uuid4().hex
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def _n_messages(self) -> int:
        snap = self.graph.get_state(self.config)
        return len(snap.values.get("messages", [])) if snap and snap.values else 0

    def _run(self, payload: Any) -> StepResult:
        before = self._n_messages()
        out = self.graph.invoke(payload, self.config)
        interrupts = out.get("__interrupt__") or []
        state = {k: v for k, v in out.items() if k != "__interrupt__"}
        return StepResult(
            state=state,
            pending_review=interrupts[0].value if interrupts else None,
            new_messages=state.get("messages", [])[before:],
        )

    def send(self, request: str, mode: Optional[Mode] = None) -> StepResult:
        if self.pending_review() is not None:
            raise RuntimeError("a review is pending; call resume() first")
        return self._run({"request": request, "requested_mode": mode})

    def resume(self, action: str, feedback: str = "") -> StepResult:
        return self._run(Command(resume={"action": action, "feedback": feedback}))

    def pending_review(self) -> Optional[dict]:
        snap = self.graph.get_state(self.config)
        for task in snap.tasks if snap else ():
            if task.interrupts:
                return task.interrupts[0].value
        return None
