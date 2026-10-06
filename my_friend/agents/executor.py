from __future__ import annotations

from typing import Dict, List, Tuple

from my_friend.graph.state import GraphState
from my_friend.tools.code_patch import PatchError, extract_symbol, outline, replace_symbol

from . import prompts
from .base import AgentDeps, assistant_turn, compose, render_files, render_history, render_spec, section
from .schemas import CodeBundle, PatchSet



def python_only(files: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """생성물은 .py만 세션에 남긴다. README/notes 같은 문서는 검증되지 않고 이후 모든 프롬프트 토큰만 늘린다
    (설명은 CodeBundle.notes / PatchSet.notes로)."""
    kept = {p: src for p, src in files.items() if p.endswith(".py")}
    return kept, sorted(p for p in files if p not in kept)

class Executor:
    """Full build: 승인된 ModelSpec → model.py + lit_module.py 전체 생성."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: GraphState) -> dict:
        bundle = self.deps.llm_for("executor").parse(
            prompts.EXECUTOR,
            compose(
                section("spec", render_spec(state)),
                section("conversation", render_history(state, self.deps.history_turns)),
                section("request", state["request"]),
            ),
            CodeBundle,
            effort=self.deps.effort["executor"],
            max_tokens=64000,
        )
        new_files, dropped = python_only({f.path: f.content for f in bundle.files})
        if dropped:
            bundle.notes += f"\n(.py 외 파일 제외: {', '.join(dropped)})"
        entry = bundle.entry_file
        if entry not in new_files:
            # 계약 위반이지만 그래프를 죽이지 않는다: validator의 import 단계 실패 → reflector가 수습
            entry = "model.py" if "model.py" in new_files else next(iter(sorted(new_files)), entry)
            bundle.notes += f"\n(entry_file '{bundle.entry_file}' 누락 → '{entry}'로 대체)"
        # 전체 재생성: 이전 빌드에만 있던 파일은 삭제("" = delete, merge_files 규약)
        stale = {p: "" for p in (state.get("files") or {}) if p not in new_files}
        listing = "\n".join(f"- {p} ({src.count(chr(10)) + 1} lines)" for p, src in new_files.items())
        return {
            "files": {**stale, **new_files},
            "entry_file": entry,
            "messages": [assistant_turn("executor", f"{bundle.notes}\n{listing}")],
        }


class PartialExecutor:
    """Sparring/Reflector가 합의한 ChangeRequest를 심볼 단위로 적용.

    LLM에는 전체 파일 대신 (모든 파일 outline + 대상 심볼 원문)만 준다 →
    입력 토큰이 파일 크기가 아니라 수정 범위에 비례.
    """

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def _locate(self, files: Dict[str, str], symbols: List[str]) -> str:
        blocks = []
        for sym in symbols:
            hits = []
            for path, src in files.items():
                if not path.endswith(".py"):
                    continue
                try:
                    code = extract_symbol(src, sym)
                except (SyntaxError, PatchError):
                    code = None
                if code:
                    hits.append(f'<symbol name="{sym}" file="{path}">\n{code}\n</symbol>')
            blocks.extend(hits or [f'<symbol name="{sym}" status="not found — create it"/>'])
        return "\n".join(blocks)

    def __call__(self, state: GraphState) -> dict:
        change = state.get("change")
        if change is None:
            return {"messages": [assistant_turn("partial_executor", "적용할 변경이 없습니다.")]}
        files = dict(state.get("files") or {})
        outlines = "\n".join(
            f"{p}:\n  " + "\n  ".join(outline(s)) for p, s in sorted(files.items()) if p.endswith(".py")
        )
        user = compose(
            section("change_request", change.model_dump_json(indent=1)),
            section("file_outlines", outlines),
            section("target_sources", self._locate(files, change.target_symbols)),
            section("spec", render_spec(state)),
            section("last_validation_error", (state.get("validation").error or "") if state.get("validation") else ""),
            # 대상 심볼을 못 찾았거나 파일이 작으면 전체 코드도 제공
            section("code", render_files(files, self.deps.max_file_chars)) if not files or len(files) <= 2 else "",
        )
        patchset = self.deps.llm_for("partial_executor").parse(
            prompts.PARTIAL_EXECUTOR, user, PatchSet,
            effort=self.deps.effort["partial_executor"], max_tokens=32000,
        )

        changed, dropped = python_only({f.path: f.content for f in patchset.new_files})
        applied, failed = [], []
        for p in patchset.patches:
            src = changed.get(p.file, files.get(p.file, ""))
            try:
                changed[p.file] = replace_symbol(src, p.symbol, p.new_source)
                applied.append(f"{p.file}:{p.symbol}")
            except (PatchError, SyntaxError) as e:
                failed.append(f"{p.file}:{p.symbol} — {e}")
        msg = f"{patchset.notes}\n적용: {', '.join(applied) or '-'}"
        if dropped:
            msg += f"\n(.py 외 파일 제외: {', '.join(dropped)})"
        if failed:
            msg += "\n실패: " + "; ".join(failed)
        return {"files": changed, "messages": [assistant_turn("partial_executor", msg)]}
