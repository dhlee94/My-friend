"""AST 기반 심볼 단위 패치 (partial update).

unified diff 대신 심볼 교체를 쓰는 이유: LLM이 만든 diff는 줄 번호/컨텍스트가 조금만 어긋나도
적용이 실패한다. 심볼 이름 + 전체 소스 교체는 위치 추정이 필요 없고, 결과를 ast.parse로 검증한다.
"""
from __future__ import annotations

import ast
import textwrap
from typing import List, Optional, Tuple


class PatchError(ValueError):
    pass


def _find(tree: ast.Module, symbol: str) -> Tuple[Optional[ast.AST], Optional[ast.ClassDef]]:
    """'Cls.method' → (method node | None, class node). 'name' → (node | None, None)."""
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    head, _, attr = symbol.partition(".")
    top = next((n for n in tree.body if isinstance(n, defs) and n.name == head), None)
    if not attr:
        return top, None
    if not isinstance(top, ast.ClassDef):
        raise PatchError(f"class '{head}' not found for '{symbol}'")
    meth = next((n for n in top.body if isinstance(n, defs) and n.name == attr), None)
    return meth, top


def _span(node: ast.AST) -> Tuple[int, int]:
    """데코레이터 포함 0-based [start, end) 라인 범위."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
    return start, node.end_lineno  # type: ignore[attr-defined]


def outline(source: str) -> List[str]:
    """LLM 컨텍스트용 심볼 목록 + 시그니처 (전체 파일 대신 전달 → 토큰 절약)."""
    tree = ast.parse(source)
    out: List[str] = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            out.append(f"class {n.name}  (L{n.lineno}-{n.end_lineno})")
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(f"    def {n.name}.{m.name}({ast.unparse(m.args)})")
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {n.name}({ast.unparse(n.args)})  (L{n.lineno}-{n.end_lineno})")
    return out


def extract_symbol(source: str, symbol: str) -> Optional[str]:
    node, _ = _find(ast.parse(source), symbol)
    if node is None:
        return None
    start, end = _span(node)
    return textwrap.dedent("\n".join(source.splitlines()[start:end]))


def replace_symbol(source: str, symbol: str, new_source: str) -> str:
    """심볼 교체. 없으면 생성(top-level은 파일 끝, 메서드는 클래스 끝)."""
    tree = ast.parse(source)
    node, cls = _find(tree, symbol)
    lines = source.splitlines()
    body = textwrap.dedent(new_source).strip("\n")

    if node is not None:
        start, end = _span(node)
        indent = " " * node.col_offset
        new_lines = textwrap.indent(body, indent, lambda l: l.strip() != "").splitlines()
        lines[start:end] = new_lines
    elif cls is not None:
        indent = " " * (cls.body[0].col_offset if cls.body else cls.col_offset + 4)
        _, end = _span(cls)
        lines[end:end] = [""] + textwrap.indent(body, indent, lambda l: l.strip() != "").splitlines()
    else:
        lines += ["", ""] + body.splitlines()

    result = "\n".join(lines) + "\n"
    try:
        ast.parse(result)
    except SyntaxError as e:
        raise PatchError(f"patch for '{symbol}' produced invalid Python: {e}") from e
    return result
