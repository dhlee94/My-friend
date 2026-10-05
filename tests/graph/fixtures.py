from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Type, Union

from pydantic import BaseModel

from my_friend.agents.schemas import DimBinding, ModelSpec, ModuleSpec, TensorSpec

CAUSAL_FORWARD = '''def forward(self, x):                      # (B, T, F)
    h = self.proj(x)                           # (B, T, D)
    h = h.transpose(1, 2)                      # (B, D, T)
    h = F.pad(h, (self.kernel - 1, 0))         # (B, D, T+k-1)  left pad only → causal
    h = self.conv(h)                           # (B, D, T)
    return self.head(h.transpose(1, 2))        # (B, T, 1)
'''

LEAKY_FORWARD = CAUSAL_FORWARD.replace(
    "F.pad(h, (self.kernel - 1, 0))         # (B, D, T+k-1)  left pad only → causal",
    "F.pad(h, ((self.kernel - 1) // 2, (self.kernel - 1) // 2))  # centered → sees future",
)

MODEL_TEMPLATE = '''import torch
import torch.nn.functional as F
from torch import nn


class TinyTS(nn.Module):
    def __init__(self, d_in, d_model, kernel):
        super().__init__()
        self.kernel = kernel
        self.proj = nn.Linear(d_in, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel)
        self.head = nn.Linear(d_model, 1)

{forward}
'''


def model_source(forward: str) -> str:
    return MODEL_TEMPLATE.format(forward="\n".join("    " + l if l else l for l in forward.splitlines()))


def tiny_spec(**over) -> ModelSpec:
    base = dict(
        name="TinyTS",
        summary="causal conv forecaster",
        dims=[DimBinding(symbol="B", value=2, meaning="batch"), DimBinding(symbol="T", value=16, meaning="time"),
              DimBinding(symbol="F", value=3, meaning="features"), DimBinding(symbol="D", value=8, meaning="width")],
        inputs=[TensorSpec(name="x", shape=["B", "T", "F"])],
        outputs=[TensorSpec(name="y", shape=["B", "T", "1"])],
        modules=[
            ModuleSpec(path="proj", role="in proj", input_shape=["B", "T", "F"], output_shape=["B", "T", "D"], complexity="O(BTFD)"),
            ModuleSpec(path="conv", role="causal conv", input_shape=["B", "D", "T+2"], output_shape=["B", "D", "T"], complexity="O(BTD²k)"),
            ModuleSpec(path="head", role="head", input_shape=["B", "T", "D"], output_shape=["B", "T", "1"], complexity="O(BTD)"),
        ],
        entry_class="TinyTS",
        init_kwargs_json='{"d_in": 3, "d_model": 8, "kernel": 3}',
        causal_axis="T",
        train_batch_size=256,
        shape_diagram="x(B,T,F) -> proj -> (B,T,D) -> conv -> (B,D,T) -> head -> (B,T,1)",
        design_decisions=["left padding for causality"],
        open_questions=[],
    )
    base.update(over)
    return ModelSpec(**base)


Response = Union[BaseModel, Callable[[str], BaseModel]]


class ScriptedLLM:
    """output_model 타입별 응답 큐. 호출 기록(user 프롬프트)을 남겨 컨텍스트 전달을 검증한다."""

    def __init__(self, script: Dict[Type[BaseModel], List[Response]]) -> None:
        self.queues: Dict[type, Deque[Response]] = defaultdict(deque, {k: deque(v) for k, v in script.items()})
        self.calls: List[tuple] = []

    def parse(self, system: str, user: str, output_model, *, effort=None, max_tokens=None):
        self.calls.append((output_model.__name__, user, effort))
        q = self.queues[output_model]
        if not q:
            raise AssertionError(f"unexpected LLM call for {output_model.__name__}")
        r = q.popleft()
        return r(user) if callable(r) and not isinstance(r, BaseModel) else r

    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        raise AssertionError("not used")

    def remaining(self) -> Dict[str, int]:
        return {k.__name__: len(v) for k, v in self.queues.items() if v}
