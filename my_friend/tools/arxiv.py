"""arXiv Atom API 클라이언트 (표준 라이브러리만 사용).

arXiv 이용 규약: 요청 간 3초 간격 권장 → 프로세스 내 단순 rate limit.
"""
from __future__ import annotations

import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Protocol

_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    summary: str
    published: str
    authors: List[str]
    categories: List[str]

    def render(self, max_chars: int = 1200) -> str:
        return (
            f"[{self.arxiv_id}] {self.title} ({self.published[:10]}; {', '.join(self.categories[:3])})\n"
            f"{self.summary[:max_chars]}"
        )


class ArxivSearcher(Protocol):
    def search(self, query: str, max_results: int = 10) -> List[ArxivEntry]: ...


class ArxivClient:
    def __init__(self, min_interval_s: float = 3.0, timeout_s: float = 20.0) -> None:
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last = 0.0
        self._lock = threading.Lock()

    def search(self, query: str, max_results: int = 10) -> List[ArxivEntry]:
        params = urllib.parse.urlencode(
            {"search_query": query, "start": 0, "max_results": max_results,
             "sortBy": "submittedDate", "sortOrder": "descending"}
        )
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            with urllib.request.urlopen(f"{_API}?{params}", timeout=self.timeout_s) as resp:
                body = resp.read()
            self._last = time.monotonic()
        return parse_feed(body)


def parse_feed(xml_bytes: bytes) -> List[ArxivEntry]:
    root = ET.fromstring(xml_bytes)
    out: List[ArxivEntry] = []
    for e in root.findall("a:entry", _NS):
        raw_id = (e.findtext("a:id", default="", namespaces=_NS) or "").rsplit("/abs/", 1)[-1]
        out.append(
            ArxivEntry(
                arxiv_id=raw_id,
                title=" ".join((e.findtext("a:title", default="", namespaces=_NS) or "").split()),
                summary=" ".join((e.findtext("a:summary", default="", namespaces=_NS) or "").split()),
                published=e.findtext("a:published", default="", namespaces=_NS) or "",
                authors=[a.findtext("a:name", default="", namespaces=_NS) for a in e.findall("a:author", _NS)],
                categories=[c.get("term", "") for c in e.findall("a:category", _NS)],
            )
        )
    return out
