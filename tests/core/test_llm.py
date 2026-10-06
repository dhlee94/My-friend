"""모델 허용 목록 / 능력별 요청 구성 / 노드별 모델 해석. 실제 API 호출 없음."""
import pytest

from types import SimpleNamespace

from my_friend.core.llm import MODEL_CAPS, MODEL_NODES, AnthropicLLM, build_llms, models_from_env

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def kwargs(model, effort="high", max_tokens=64000):
    return AnthropicLLM(model=model, client=object()).request_kwargs("sys", "user", SCHEMA, effort, max_tokens)


def test_opus_sends_effort_thinking_and_fallback():
    kw = kwargs("claude-opus-5-5")
    assert kw["output_config"]["effort"] == "high"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]


def test_haiku_drops_unsupported_params_and_clamps_max_tokens():
    kw = kwargs("claude-haiku-4-5", max_tokens=100_000)
    assert "effort" not in kw["output_config"] and "format" in kw["output_config"]
    assert "thinking" not in kw and "fallbacks" not in kw and "betas" not in kw
    assert kw["max_tokens"] == 64_000


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-sonnet-5", "claude-fable-5"])
def test_models_without_default_fallback(model):
    kw = kwargs(model)
    assert "fallbacks" not in kw and kw["output_config"]["effort"] == "high"


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="allowed"):
        AnthropicLLM(model="claude-3-opus", client=object())


def test_models_from_env_defaults_and_overrides():
    d = models_from_env({})
    assert {n for n, m in d.items() if m == "claude-haiku-4-5"} == {"router", "paper_scout", "query_rewriter"}
    assert {m for n, m in d.items() if n not in {"router", "paper_scout", "query_rewriter"}} == {"claude-sonnet-5-5"}
    assert set(models_from_env({"MY_FRIEND_MODEL": "claude-haiku-4-5"}).values()) == {"claude-haiku-4-5"}  # 일괄 지정
    m = models_from_env({"MY_FRIEND_MODEL": "claude-sonnet-5-5", "MY_FRIEND_MODEL_ROUTER": "claude-haiku-4-5"})
    assert m["router"] == "claude-haiku-4-5"
    assert all(v == "claude-sonnet-5-5" for k, v in m.items() if k != "router")
    assert set(m) == set(MODEL_NODES)
    assert all(v in MODEL_CAPS for v in m.values())


def test_models_from_env_names_the_bad_variable():
    with pytest.raises(ValueError, match="MY_FRIEND_MODEL_EXECUTOR"):
        models_from_env({"MY_FRIEND_MODEL_EXECUTOR": "gpt-5"})
    with pytest.raises(ValueError, match="^MY_FRIEND_MODEL:"):
        models_from_env({"MY_FRIEND_MODEL": "claude-opus-5-5-20260401"})


def test_build_llms_shares_instances_per_model():
    llms = build_llms(models_from_env({"MY_FRIEND_MODEL_ROUTER": "claude-haiku-4-5"}), client=object())
    assert llms["router"].model == "claude-haiku-4-5"
    assert llms["planner"] is llms["executor"]            # 같은 모델 → 같은 인스턴스


def test_usage_recorded_with_responding_model_price():
    llm = AnthropicLLM(model="claude-sonnet-5-5", client=object())
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000,
                            cache_creation_input_tokens=0, cache_read_input_tokens=0)
    llm._record("ModelSpec", SimpleNamespace(usage=usage, model="claude-sonnet-5-5"))
    llm._record("ModelSpec", SimpleNamespace(usage=usage, model="claude-opus-4-8"))   # fallback 응답
    assert [round(u.usd, 2) for u in llm.usage_log] == [3.00, 7.50]      # 1M*$2 + 0.1M*$10 / 1M*$5 + 0.1M*$25
    assert llm.usage_log[0].label == "ModelSpec"


def test_usage_normalizes_dated_snapshot_ids():
    llm = AnthropicLLM(model="claude-haiku-4-5", client=object())
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=0,
                            cache_creation_input_tokens=0, cache_read_input_tokens=0)
    llm._record("X", SimpleNamespace(usage=usage, model="claude-haiku-4-5-20251001"))
    llm._record("X", SimpleNamespace(usage=usage, model="claude-opus-5-5-20260901"))    # opus-5 아님
    assert [u.model for u in llm.usage_log] == ["claude-haiku-4-5", "claude-opus-5-5"]
    assert [u.usd for u in llm.usage_log] == [1.0, 4.0]
