"""마크다운 → 구조 인식 청크.

파이프라인: 라인 기반 블록 파서 → (헤더 스택으로 breadcrumb 추적) → 토큰 예산 기반 패킹.

보호 블록(equation/table/code)은 절대 분할하지 않는다. 예산 초과 시 해당 블록만 단독 청크(oversized)로
내보낸다 — 수식이 반으로 잘린 청크는 임베딩도 BM25도 오염시키므로, 큰 청크 하나가 차라리 낫다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .schema import Chunk, classify_section

PROTECTED_KINDS = frozenset({"equation", "table", "code"})

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_ROW = re.compile(r"^\s*\|")
_ENV_BEGIN = re.compile(r"^\s*\\begin\{(equation|align|gather|multline|eqnarray)(\*?)\}")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_CAPTION = re.compile(r"^\s*(?:\*\*|_)?(?:Table|Tab\.)\s*[0-9IVX]+", re.I)
_INLINE_MATH = re.compile(r"\$[^$\n]+?\$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\x00])")


@dataclass
class Block:
    kind: str           # heading | paragraph | list | equation | table | code
    text: str
    level: int = 0      # heading level


def parse_blocks(markdown: str) -> List[Block]:
    lines = markdown.splitlines()
    blocks: List[Block] = []
    para: List[str] = []

    def flush_para() -> None:
        if para:
            text = "\n".join(para).strip()
            if text:
                kind = "list" if _LIST_ITEM.match(para[0]) else "paragraph"
                blocks.append(Block(kind, text))
            para.clear()

    def collect_until(start: int, is_end: Callable[[str], bool]) -> int:
        """lines[start]부터 종료 라인(포함)까지 수집. 미종결이면 EOF까지 — 잘린 블록이라도 분할은 안 한다."""
        j = start + 1
        while j < len(lines) and not is_end(lines[j]):
            j += 1
        return min(j, len(lines) - 1)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        fence = _FENCE.match(line)
        if fence:
            flush_para()
            marker = fence.group(1)
            end = collect_until(i, lambda l: l.strip().startswith(marker))
            blocks.append(Block("code", "\n".join(lines[i : end + 1])))
            i = end + 1
            continue

        if stripped.startswith("$$"):
            flush_para()
            if stripped.count("$$") >= 2:      # $$ ... $$ 한 줄
                end = i
            else:
                end = collect_until(i, lambda l: "$$" in l)
            blocks.append(Block("equation", "\n".join(lines[i : end + 1])))
            i = end + 1
            continue

        if stripped.startswith("\\["):
            flush_para()
            end = i if "\\]" in stripped else collect_until(i, lambda l: "\\]" in l)
            blocks.append(Block("equation", "\n".join(lines[i : end + 1])))
            i = end + 1
            continue

        env = _ENV_BEGIN.match(line)
        if env:
            flush_para()
            closing = f"\\end{{{env.group(1)}{env.group(2)}}}"
            end = i if closing in line else collect_until(i, lambda l: closing in l)
            blocks.append(Block("equation", "\n".join(lines[i : end + 1])))
            i = end + 1
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_para()
            blocks.append(Block("heading", heading.group(2).strip(), level=len(heading.group(1))))
            i += 1
            continue

        if _TABLE_ROW.match(line):
            flush_para()
            j = i
            while j + 1 < len(lines) and _TABLE_ROW.match(lines[j + 1]):
                j += 1
            blocks.append(Block("table", "\n".join(lines[i : j + 1])))
            i = j + 1
            continue

        if not stripped:
            flush_para()
        else:
            para.append(line)
        i += 1

    flush_para()
    return _attach_table_captions(blocks)


def _attach_table_captions(blocks: List[Block]) -> List[Block]:
    """'Table 2: ...' 캡션을 인접 표에 흡수 → 캡션과 표가 다른 청크로 갈라지는 것 방지."""
    out: List[Block] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        if b.kind == "paragraph" and _CAPTION.match(b.text) and nxt is not None and nxt.kind == "table":
            out.append(Block("table", f"{b.text}\n\n{nxt.text}"))
            i += 2
            continue
        if b.kind == "paragraph" and _CAPTION.match(b.text) and out and out[-1].kind == "table":
            out[-1] = Block("table", f"{out[-1].text}\n\n{b.text}")
            i += 1
            continue
        out.append(b)
        i += 1
    return out


def _default_token_counter(encoding: str = "cl100k_base") -> Callable[[str], int]:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda s: len(enc.encode(s, disallowed_special=()))


class MarkdownChunker:
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        max_header_level: int = 3,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.max_header_level = max_header_level
        self.count = token_counter or _default_token_counter()

    # ------------------------------------------------------------------ public
    def chunk(self, markdown: str, doc_id: str, doc_title: Optional[str] = None) -> List[Chunk]:
        chunks: List[Chunk] = []
        stack: List[Tuple[int, str]] = []          # (level, title)
        buf: List[Tuple[Block, int]] = []          # (block, tokens)
        n_carried = 0                              # buf 앞쪽의 overlap 블록 수
        title = doc_title

        def emit(oversized: bool = False) -> None:
            if len(buf) <= n_carried:              # overlap만 남은 버퍼는 버림
                return
            content = "\n\n".join(b.text for b, _ in buf)
            breadcrumb = [t for _, t in stack]
            kinds = sorted({b.kind for b, _ in buf})
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{len(chunks):04d}",
                    doc_id=doc_id,
                    doc_title=title or doc_id,
                    order=len(chunks),
                    breadcrumb=breadcrumb,
                    section_type=classify_section(breadcrumb),
                    content=content,
                    block_types=kinds,
                    token_count=self.count(content),
                    oversized=oversized,
                )
            )

        def flush(carry: bool) -> None:
            nonlocal buf, n_carried
            emit()
            if not carry:
                buf, n_carried = [], 0
                return
            # 뒤에서부터 overlap 예산 안의 블록을 다음 청크로 이월.
            # 블록 단위 overlap이라 수식 직후 "where W_q is ..." 문단이 수식 문맥을 잃지 않는다.
            kept: List[Tuple[Block, int]] = []
            budget = self.overlap_tokens
            for b, t in reversed(buf):
                if t > budget:
                    break
                kept.insert(0, (b, t))
                budget -= t
            buf, n_carried = kept, len(kept)

        for block in parse_blocks(markdown):
            if block.kind == "heading" and block.level <= self.max_header_level:
                flush(carry=False)                 # 청크는 섹션 경계를 넘지 않는다
                if title is None and block.level == 1:
                    title = block.text             # 첫 H1 = 논문 제목, breadcrumb에선 제외
                    continue
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text))
                continue
            if block.kind == "heading":            # H4+ 는 본문 텍스트로 취급
                block = Block("paragraph", "#" * block.level + " " + block.text)

            for piece in self._split_if_needed(block):
                t = self.count(piece.text)
                if buf and sum(x for _, x in buf) + t > self.max_tokens:
                    flush(carry=True)
                    if sum(x for _, x in buf) + t > self.max_tokens:
                        buf, n_carried = [], 0     # overlap을 넣으면 넘치는 경우 overlap 포기
                buf.append((piece, t))
                if t > self.max_tokens:            # 보호 블록 단독 초과
                    emit(oversized=True)
                    buf, n_carried = [], 0

        flush(carry=False)
        return chunks

    # ----------------------------------------------------------------- helpers
    def _split_if_needed(self, block: Block) -> List[Block]:
        if block.kind in PROTECTED_KINDS or self.count(block.text) <= self.max_tokens:
            return [block]
        if block.kind == "list":
            units = [u for u in re.split(r"\n(?=\s*(?:[-*+]|\d+[.)])\s+)", block.text) if u.strip()]
        else:
            units = _split_sentences(block.text)
        out: List[Block] = []
        for u in units:
            if self.count(u) <= self.max_tokens:
                out.append(Block(block.kind, u))
            else:
                out.extend(Block(block.kind, w) for w in self._pack_words(u))
        return out

    def _pack_words(self, text: str) -> List[str]:
        """문장 하나가 예산 초과일 때의 최후 수단. 인라인 수식 토큰은 마스킹돼 있어 내부가 잘리지 않는다."""
        masked, spans = _mask_inline_math(text)
        pieces, cur = [], []
        for w in masked.split():
            cand = " ".join(cur + [w])
            if cur and self.count(_unmask(cand, spans)) > self.max_tokens:
                pieces.append(_unmask(" ".join(cur), spans))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            pieces.append(_unmask(" ".join(cur), spans))
        return pieces


def _mask_inline_math(text: str) -> Tuple[str, List[str]]:
    spans: List[str] = []

    def repl(m: "re.Match[str]") -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    return _INLINE_MATH.sub(repl, text), spans


def _unmask(text: str, spans: List[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def _split_sentences(text: str) -> List[str]:
    """인라인 수식($a. b$)을 마스킹한 뒤 문장 분리 → 수식 내부 마침표에서 잘리지 않음."""
    masked, spans = _mask_inline_math(text)
    return [_unmask(s, spans) for s in _SENT_SPLIT.split(masked) if s.strip()]
