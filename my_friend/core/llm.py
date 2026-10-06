"""LLM 클라이언트 경계.

에이전트/RAG는 `StructuredLLM` 프로토콜에만 의존한다 → 테스트에서는 fake, 운영에서는 Claude.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Type, TypeVar

import anthropic
from pydantic import BaseModel

DEFAULT_MODEL = "claude-opus-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class ModelCaps:
    effort: bool            # output_config.effort 지원 (Haiku 4.5는 400)
    fallback: bool          # fallbacks="default" 서버측 거절 재시도 지원
    thinking: bool          # thinking={"type": "adaptive"} 지원 (Opus 4.8은 생략 시 thinking 꺼짐 → 명시)
    max_output: int


# 허용 모델 = structured outputs(output_config.format) 지원 모델. 모든 노드가 구조화 출력에 의존한다.
# 목록에 없는 모델은 시작 시점에 거부한다 (요청 시점 400보다 빨리, 어떤 설정이 문제인지 명확히).
MODEL_CAPS: Dict[str, ModelCaps] = {
    "claude-fable-5-1":  ModelCaps(effort=True,  fallback=True,  thinking=True,  max_output=128_000),
    "claude-fable-5":    ModelCaps(effort=True,  fallback=False, thinking=True,  max_output=128_000),
    "claude-opus-5-5":   ModelCaps(effort=True,  fallback=True,  thinking=True,  max_output=128_000),
    "claude-opus-5":     ModelCaps(effort=True,  fallback=True,  thinking=True,  max_output=128_000),
    "claude-opus-4-8":   ModelCaps(effort=True,  fallback=False, thinking=True,  max_output=128_000),
    "claude-sonnet-5-5": ModelCaps(effort=True,  fallback=True,  thinking=True,  max_output=128_000),
    "claude-sonnet-5":   ModelCaps(effort=True,  fallback=False, thinking=True,  max_output=128_000),
    "claude-haiku-4-5":  ModelCaps(effort=False, fallback=False, thinking=False, max_output=64_000),
}


def model_caps(model: str) -> ModelCaps:
    try:
        return MODEL_CAPS[model]
    except KeyError:
        raise ValueError(f"unsupported model {model!r}; allowed: {', '.join(MODEL_CAPS)}") from None


# 모델을 따로 지정할 수 있는 단위: 그래프 노드 + RAG 쿼리 재작성기
MODEL_NODES: Tuple[str, ...] = (
    "router", "paper_scout", "sparring_partner", "planner",
    "executor", "partial_executor", "reflector", "query_rewriter",
)


def models_from_env(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """노드별 모델: MY_FRIEND_MODEL_<NODE> > MY_FRIEND_MODEL > DEFAULT_MODEL. 허용 목록 밖이면 즉시 실패."""
    env = os.environ if env is None else env
    default = env.get("MY_FRIEND_MODEL") or DEFAULT_MODEL
    models = {}
    for node in MODEL_NODES:
        var = f"MY_FRIEND_MODEL_{node.upper()}"
        model = env.get(var) or default
        try:
            model_caps(model)
        except ValueError as e:
            raise ValueError(f"{var if env.get(var) else 'MY_FRIEND_MODEL'}: {e}") from None
        models[node] = model
    return models

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
    - 모델 능력(MODEL_CAPS)에 맞춰 effort / thinking / fallbacks를 넣거나 뺀다.
    - 안전 분류기 거절 시 `fallbacks="default"`로 서버측 재시도 (지원 모델만, beta).
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
        self.caps = model_caps(model)
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()

    def request_kwargs(self, system: str, user: str, schema: Dict[str, Any], effort: str, max_tokens: int) -> dict:
        output_config: Dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.caps.effort:
            output_config["effort"] = effort
        kw: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=min(max_tokens, self.caps.max_output),
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        if self.caps.thinking:
            kw["thinking"] = {"type": "adaptive"}
        if self.caps.fallback:
            kw.update(betas=[_FALLBACK_BETA], fallbacks="default")
        return kw

    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.beta.messages.create(
            **self.request_kwargs(system, user, schema, self.effort, self.max_tokens)
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
        kw = self.request_kwargs(system, user, schema, effort or self.effort, max_tokens or 64000)
        with self.client.beta.messages.stream(**kw) as stream:
            response = stream.get_final_message()
        return output_model.model_validate_json(_final_text(response))


def build_llms(models: Mapping[str, str], client: Optional[anthropic.Anthropic] = None) -> Dict[str, AnthropicLLM]:
    """노드→모델 맵을 노드→AnthropicLLM으로. 같은 모델은 인스턴스를, 모든 모델은 HTTP 클라이언트를 공유한다."""
    client = client or anthropic.Anthropic()
    by_model = {m: AnthropicLLM(model=m, client=client) for m in sorted(set(models.values()))}
    return {node: by_model[m] for node, m in models.items()}


def _final_text(response: Any) -> str:
    if response.stop_reason == "refusal":
        raise LLMRefusalError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise LLMTruncatedError("structured output truncated by max_tokens")
    return next(b.text for b in response.content if b.type == "text")
