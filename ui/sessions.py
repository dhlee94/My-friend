"""이전 세션 목록 (Chainlit 비의존).

세션 복원의 단일 원천은 LangGraph 체크포인트다: Chainlit data layer/인증 없이 체크포인트에서
스레드 목록을 뽑고, 선택된 스레드는 체크포인트 state로 대화·다이어그램을 다시 그린다 (#11).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# 체크포인트 스캔 상한 (SQL 경로가 없는 saver용): 스레드 하나가 수십~수백 superstep을 남길 수 있다
_SCAN_LIMIT = 2000


@dataclass(frozen=True)
class SessionInfo:
    thread_id: str
    title: str              # 첫 사용자 발화 (잘라서)
    updated_at: str         # 마지막 체크포인트 시각 (ISO 8601, UTC)
    pending_review: bool    # 승인 대기 중에 멈춘 세션

    @property
    def label(self) -> str:
        try:
            when = datetime.fromisoformat(self.updated_at).astimezone().strftime("%m/%d %H:%M")   # 로컬 시각
        except ValueError:
            when = "?"
        return f"{'⏸ ' if self.pending_review else ''}{when} · {self.title}"


async def _recent_thread_ids(checkpointer, limit: int, exclude: Optional[str]) -> List[str]:
    if type(checkpointer).__name__ == "AsyncSqliteSaver":
        # checkpoint_id는 시간순 정렬되는 uuid6 → 스레드별 최신 id로 정렬. 블롭 역직렬화 없음.
        async with checkpointer.conn.execute(
            "SELECT thread_id FROM checkpoints WHERE checkpoint_ns = '' GROUP BY thread_id "
            "ORDER BY MAX(checkpoint_id) DESC LIMIT ?",
            (limit + 1,),
        ) as cur:
            rows = [r[0] for r in await cur.fetchall()]
    else:
        # 일반 saver: alist(None)의 순서는 구현마다 다르다(MemorySaver는 스레드 단위 순회) → 스레드별 최신 id로 직접 정렬
        latest = {}
        n = 0
        async for tup in checkpointer.alist(None):
            tid, cid = tup.config["configurable"]["thread_id"], tup.config["configurable"]["checkpoint_id"]
            latest[tid] = max(latest.get(tid, ""), cid)
            n += 1
            if n >= _SCAN_LIMIT:
                break
        rows = sorted(latest, key=latest.get, reverse=True)[: limit + 1]
    return [t for t in rows if t != exclude][:limit]


async def recent_sessions(graph, limit: int = 5, exclude: Optional[str] = None, title_chars: int = 40) -> List[SessionInfo]:
    """최근 세션 (최신순). 사용자 발화가 없는 빈 스레드는 제외."""
    out: List[SessionInfo] = []
    for tid in await _recent_thread_ids(graph.checkpointer, limit, exclude):
        snap = await graph.aget_state({"configurable": {"thread_id": tid}})
        messages = (snap.values or {}).get("messages") or []
        first = next((m["content"] for m in messages if m.get("role") == "user"), None)
        if not first:
            continue
        title = " ".join(first.split())
        out.append(SessionInfo(
            thread_id=tid,
            title=title if len(title) <= title_chars else title[: title_chars - 1] + "…",
            updated_at=snap.created_at or "",
            pending_review=any(t.interrupts for t in snap.tasks),
        ))
    return out
