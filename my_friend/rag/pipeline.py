"""오프라인 인덱싱 / 검색 CLI.

    python -m my_friend.rag.pipeline build  --input data/papers --out data/index
    python -m my_friend.rag.pipeline search --index data/index "RoPE 구현에서 주파수 계산 방식" --section methodology
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from .analyzer import EnglishAnalyzer
from .chunker import MarkdownChunker
from .embedder import Embedder
from .index import HybridIndex
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


def _main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="my_friend.rag.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--input", type=Path, required=True, help="*.md 디렉토리")
    b.add_argument("--out", type=Path, required=True)
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

    if args.cmd == "build":
        embedder = SentenceTransformerEmbedder(args.model)
        index = build_index(
            args.input.rglob("*.md"), embedder, MarkdownChunker(args.max_tokens, args.overlap)
        )
        index.save(args.out)
        n_over = sum(c.oversized for c in index.chunks)
        print(f"indexed {len(index.chunks)} chunks ({n_over} oversized protected blocks) → {args.out}")
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
