"""오프라인 인덱싱 / 검색 CLI.

    python -m my_friend.rag.pipeline sync   --input data/papers --out data/index [--gc]
    python -m my_friend.rag.pipeline search --index data/index "RoPE 구현에서 주파수 계산 방식" --section methodology

sync = 입력 폴더를 정답으로 인덱스를 맞춘다 (추가/수정/삭제 모두). 청킹·키워드 IDF·BM25는 매번 코퍼스 전체로
재계산하고(밀리초), 임베딩만 로컬 캐시(data/embed_cache.sqlite)에서 재사용한다 → 증분이어도 전체 재빌드와
같은 인덱스가 나온다 (IDF drift 없음, #5). `build`는 sync의 별칭.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .analyzer import EnglishAnalyzer
from .chunker import MarkdownChunker
from .embedder import Embedder
from .embed_cache import CachedEmbedder, EmbeddingCache
from .index import HybridIndex, read_sources
from .keywords import KeywordExtractor
from .schema import SECTION_TYPES, Chunk, SearchFilter


def chunk_documents(paths: Iterable[Path], chunker: MarkdownChunker) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(paths):
        chunks.extend(chunker.chunk(path.read_text(encoding="utf-8"), doc_id=path.stem))
    return chunks


def build_index(
    paths: Iterable[Path],
    embedder: Embedder,
    chunker: Optional[MarkdownChunker] = None,
    analyzer: Optional[EnglishAnalyzer] = None,
    top_k_keywords: int = 8,
) -> HybridIndex:
    analyzer = analyzer or EnglishAnalyzer()
    chunks = chunk_documents(paths, chunker or MarkdownChunker())
    # 2-pass: 키워드 idf는 코퍼스 전체 통계가 필요하므로 청킹 완료 후 일괄 계산
    KeywordExtractor(analyzer, top_k=top_k_keywords).assign(chunks)
    return HybridIndex.build(chunks, embedder, analyzer)


@dataclass
class SyncReport:
    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    n_chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0          # 새로 임베딩한 고유 텍스트 수
    gc_deleted: int = 0

    def summary(self) -> str:
        def ids(xs: List[str]) -> str:
            return f" ({', '.join(xs[:5])}{' …' if len(xs) > 5 else ''})" if xs else ""
        return (f"docs: +{len(self.added)}{ids(self.added)} ~{len(self.modified)}{ids(self.modified)} "
                f"-{len(self.removed)}{ids(self.removed)} ={len(self.unchanged)}\n"
                f"chunks: {self.n_chunks} | embeddings: {self.cache_hits} cached, {self.cache_misses} computed"
                + (f" | gc: {self.gc_deleted} stale vectors removed" if self.gc_deleted else ""))


def _swap_dir(new: Path, out: Path) -> None:
    """완성된 새 인덱스로 교체. 실패해도 기존 인덱스는 남는다 (중간 산출물은 new 쪽에만 있다)."""
    old = out.with_name(out.name + ".sync-old")
    if old.exists():
        shutil.rmtree(old)
    if out.exists():
        os.replace(out, old)
    os.replace(new, out)
    if old.exists():
        shutil.rmtree(old)


def sync_index(
    input_dir: Path,
    out: Path,
    embedder: Embedder,
    cache_path: Path,
    chunker: Optional[MarkdownChunker] = None,
    analyzer: Optional[EnglishAnalyzer] = None,
    gc: bool = False,
) -> SyncReport:
    """input_dir의 *.md 전체로 인덱스를 다시 맞춘다. 같은 입력이면 몇 번을 돌려도 같은 인덱스."""
    out = Path(out)
    paths = sorted(Path(input_dir).rglob("*.md"))
    if not paths:
        raise ValueError(f"no *.md files under {input_dir}")
    sources: Dict[str, str] = {}
    for p in paths:                                      # doc_id = 파일 stem → 하위 폴더 간 이름 충돌 금지
        if p.stem in sources:
            raise ValueError(f"duplicate doc id '{p.stem}' (file names must be unique across subfolders)")
        sources[p.stem] = hashlib.sha256(p.read_bytes()).hexdigest()

    old = read_sources(out)
    rep = SyncReport(
        added=sorted(d for d in sources if d not in old),
        modified=sorted(d for d in sources if d in old and old[d] != sources[d]),
        removed=sorted(d for d in old if d not in sources),
        unchanged=sorted(d for d in sources if old.get(d) == sources[d]),
    )
    cache = EmbeddingCache(cache_path)
    try:
        cached = CachedEmbedder(embedder, cache)
        index = build_index(paths, cached, chunker, analyzer)
        tmp = out.with_name(out.name + ".sync-tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        index.save(tmp, sources=sources)
        _swap_dir(tmp, out)
        rep.n_chunks, rep.cache_hits, rep.cache_misses = len(index.chunks), cached.hits, cached.misses
        if gc:
            rep.gc_deleted = cache.gc(set(cached.last_keys))
    finally:
        cache.close()
    return rep


def _main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="my_friend.rag.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("sync", aliases=["build"], help="입력 폴더 기준으로 인덱스 생성/갱신 (추가·수정·삭제)")
    b.add_argument("--input", type=Path, required=True, help="*.md 디렉토리 (하위 폴더 포함)")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--cache", type=Path, help="임베딩 캐시 (기본: <out의 부모>/embed_cache.sqlite)")
    b.add_argument("--gc", action="store_true", help="현재 인덱스가 쓰지 않는 캐시 벡터 삭제")
    b.add_argument("--model", default="BAAI/bge-m3")
    b.add_argument("--max-tokens", type=int, default=768)
    b.add_argument("--overlap", type=int, default=128)

    s = sub.add_parser("search")
    s.add_argument("--index", type=Path, required=True)
    s.add_argument("query")
    s.add_argument("--section", choices=SECTION_TYPES)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--no-rewrite", action="store_true")

    args = ap.parse_args(argv)
    from .embedder import SentenceTransformerEmbedder

    if args.cmd in ("sync", "build"):
        rep = sync_index(
            args.input, args.out, SentenceTransformerEmbedder(args.model),
            cache_path=args.cache or args.out.parent / "embed_cache.sqlite",
            chunker=MarkdownChunker(args.max_tokens, args.overlap), gc=args.gc,
        )
        print(f"synced → {args.out}\n{rep.summary()}")
        return

    from .query_rewriter import QueryRewriter
    from .retriever import HybridRetriever

    index = HybridIndex.load(args.index)
    rewriter = None
    if not args.no_rewrite:
        from my_friend.core.llm import AnthropicLLM, models_from_env

        rewriter = QueryRewriter(AnthropicLLM(model=models_from_env()["query_rewriter"]))
    retriever = HybridRetriever(index, SentenceTransformerEmbedder(index.embedder_name), rewriter)
    flt = SearchFilter(section_types=frozenset({args.section})) if args.section else None
    res = retriever.retrieve(args.query, top_k=args.top_k, flt=flt)

    print(f"query    : {res.query.original}\nrewritten: {res.query.english_query}\nkeywords : {res.query.keywords}")
    print(f"filter   : {res.applied_filter}\n" + "".join(f"note     : {n}\n" for n in res.notes))
    for rank, h in enumerate(res.hits, 1):
        print(f"\n#{rank} rrf={h.score:.4f} lex#{h.lexical_rank} dense#{h.dense_rank}  {h.chunk.breadcrumb_str}")
        print(f"   [{h.chunk.section_type}] blocks={h.chunk.block_types} kw={h.chunk.keyword_weights}")
        print("   " + h.chunk.content[:300].replace("\n", "\n   "))


if __name__ == "__main__":
    _main()
