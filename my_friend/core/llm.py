"""LLM 클라이언트 경계.

에이전트/RAG는 `StructuredLLM` 프로토콜에만 의존한다 → 테스트에서는 fake, 운영에서는 Claude.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple, Type, TypeVar

import anthropic
from pydantic import BaseModel

# 비용 상한 = Sonnet 5.5 (2026-10-06 결정). 설계·코드·디버깅 노드는 Sonnet 5.5,
# 분류·검색 계획·요약·쿼리 재작성처럼 가벼운 노드는 Haiku 4.5. Opus/Fable은 .env로만 명시적으로 켠다.
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_NODE_MODELS: Dict[str, str] = {
    "router": "claude-haiku-4-5",
    "paper_scout": "claude-haiku-4-5",
    "query_rewriter": "claude-haiku-4-5",
}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class ModelCaps:
    effort: bool            # output_config.effort 지원 (Haiku 4.5는 400)
    fallback: bool          # fallbacks="default" 서버측 거절 재시도 지원
    thinking: bool          # thinking={"type": "adaptive"} 지원 (Opus 4.8은 생략 시 thinking 꺼짐 → 명시)
    max_output: int
    usd_in: float           # $ / 1M input tokens  (캐시 쓰기 1.25x, 캐시 읽기 0.1x)
    usd_out: float          # $ / 1M output tokens (thinking 포함)


# 허용 모델 = structured outputs(output_config.format) 지원 모델. 모든 노드가 구조화 출력에 의존한다.
# 목록에 없는 모델은 시작 시점에 거부한다 (요청 시점 400보다 빨리, 어떤 설정이 문제인지 명확히).
MODEL_CAPS: Dict[str, ModelCaps] = {
    "claude-fable-5-1":  ModelCaps(effort=True,  fallback=True,  thinking=True,  max_output=128_000, usd_in=10.00, usd_out=50.00),
    "claude-fable-5":    ModelCaps(effort=True,  fallback=False, thinking=True,  max_output=128_000, usd_in=10.00, usd_out=50.00),
    "claude-opus-5-5":   ModelCaps(effort=True,  fallback=True,  thinking=True,  max_output=128_000, usd_in=4.00, usd_out=20.00),
    "claude-opus-5":     ModelCaps(effort=True,  fallback=True,  thinking=True,  max_output=128_000, usd_in=5.00, usd_out=25.00),
    "claude-opus-4-8":   ModelCaps(effort=True,  fallback=False, thinking=True,  max_output=128_000, usd_in=5.00, usd_out=25.00),
    "claude-sonnet-5-5": ModelCaps(effort=True,  fallback=True,  thinking=True,  max_output=128_000, usd_in=2.00, usd_out=10.00),
    "claude-sonnet-5":   ModelCaps(effort=True,  fallback=False, thinking=True,  max_output=128_000, usd_in=2.00, usd_out=10.00),
    "claude-haiku-4-5":  ModelCaps(effort=False, fallback=False, thinking=False, max_output=64_000, usd_in=1.00, usd_out=5.00),
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
    """노드별 모델: MY_FRIEND_MODEL_<NODE> > MY_FRIEND_MODEL(전 노드 일괄) > DEFAULT_NODE_MODELS > DEFAULT_MODEL.
    허용 목록 밖이면 즉시 실패."""
    env = os.environ if env is None else env
    models = {}
    for node in MODEL_NODES:
        var = f"MY_FRIEND_MODEL_{node.upper()}"
        model = env.get(var) or env.get("MY_FRIEND_MODEL") or DEFAULT_NODE_MODELS.get(node, DEFAULT_MODEL)
        try:
            model_caps(model)
        except ValueError as e:
            raise ValueError(f"{var if env.get(var) else 'MY_FRIEND_MODEL'}: {e}") from None
        models[node] = model
    return models

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Usage:
    """호출 1회의 토큰/비용. label = 출력 스키마 이름(RouteDecision, ModelSpec, ...) → 노드 식별용."""

    model: str
    label: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    usd: float


def usage_cost(caps: ModelCaps, inp: int, out: int, cache_write: int = 0, cache_read: int = 0) -> float:
    return (inp * caps.usd_in + cache_write * caps.usd_in * 1.25 + cache_read * caps.usd_in * 0.1
            + out * caps.usd_out) / 1e6


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
        self.usage_log: List[Usage] = []     # 호출마다 append (같은 모델 노드끼리 인스턴스 공유 → label로 구분)

    def _record(self, label: str, response: Any) -> None:
        u = getattr(response, "usage", None)
        if u is None:
            return
        inp, out = u.input_tokens or 0, u.output_tokens or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        # fallback으로 다른 모델이 응답했으면 그 모델 단가로. response.model은 날짜 스냅샷 ID일 수 있다
        # (claude-haiku-4-5-20251001) → 허용 목록의 별칭으로 정규화
        model = _alias(getattr(response, "model", None) or self.model)
        caps = MODEL_CAPS.get(model, self.caps)
        self.usage_log.append(Usage(model, label, inp, out, cw, cr, usage_cost(caps, inp, out, cw, cr)))

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
        self._record("generate_json", response)
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
        self._record(output_model.__name__, response)     # 거절/잘림도 과금되므로 검사 전에 기록
        return output_model.model_validate_json(_final_text(response))


def _alias(model: str) -> str:
    """날짜 스냅샷 ID → MODEL_CAPS 별칭. 가장 긴 접두사 매칭 (claude-opus-5-5-… 가 claude-opus-5 로 잡히지 않게)."""
    if model in MODEL_CAPS:
        return model
    hits = [k for k in MODEL_CAPS if model.startswith(k + "-")]
    return max(hits, key=len) if hits else model


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
