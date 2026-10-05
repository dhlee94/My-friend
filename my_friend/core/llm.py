"""LLM 클라이언트 경계.

에이전트/RAG는 `StructuredLLM` 프로토콜에만 의존한다 → 테스트에서는 fake, 운영에서는 Claude.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Protocol, Type, TypeVar

import anthropic
from pydantic import BaseModel

DEFAULT_MODEL = "claude-opus-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class LLMRefusalError(RuntimeError):
    pass


class LLMTruncatedError(RuntimeError):
    pass


class StructuredLLM(Protocol):
    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """`schema`(JSON Schema)를 만족하는 dict를 반환."""
        ...

    def parse(
        self,
        system: str,
        user: str,
        output_model: Type[T],
        *,
        effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        """pydantic 모델로 검증된 구조화 출력."""
        ...


class AnthropicLLM:
    """Claude structured outputs 기반 생성기.

    - `output_config.format`으로 스키마 강제 → 파싱 실패 재시도 루프 불필요.
    - 안전 분류기 거절 시 `fallbacks="default"`로 서버측 재시도 (beta).
    - `parse()`는 스트리밍: 코드 생성처럼 출력이 긴 노드에서 HTTP 타임아웃 회피.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        max_tokens: int = 8000,
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()

    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            betas=[_FALLBACK_BETA],
            fallbacks="default",
        )
        return json.loads(_final_text(response))

    def parse(
        self,
        system: str,
        user: str,
        output_model: Type[T],
        *,
        effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        schema = anthropic.transform_schema(output_model.model_json_schema())
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=max_tokens or 64000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": effort or self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            betas=[_FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
        return output_model.model_validate_json(_final_text(response))


def _final_text(response: Any) -> str:
    if response.stop_reason == "refusal":
        raise LLMRefusalError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise LLMTruncatedError("structured output truncated by max_tokens")
    return next(b.text for b in response.content if b.type == "text")
