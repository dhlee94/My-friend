"""노드 간 계약(contract). LLM 구조화 출력 스키마이자 그래프 상태에 저장되는 값.

규칙: 임의 키 dict(dict[str, X])는 쓰지 않는다 — structured outputs가 additionalProperties=false를
요구하므로 list[Binding] 형태로 표현한다.
"""
from __future__ import annotations

import json
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["sparring", "full_build"]


# ---------------------------------------------------------------- router
class RouteDecision(BaseModel):
    mode: Mode
    reason: str


# ----------------------------------------------------------- paper_scout
class ScoutSearchPlan(BaseModel):
    arxiv_query: str = Field(description='arXiv API search_query, e.g. abs:"state space model" AND cat:cs.LG')
    local_query: str = Field(description="English query for the local paper/code RAG index")


class PaperSummary(BaseModel):
    arxiv_id: str
    title: str
    published: str
    key_idea: str
    relevance: str


class ScoutReport(BaseModel):
    papers: List[PaperSummary]
    trends: List[str]
    digest: str = Field(description="Korean summary for the sparring partner and the user")


# ------------------------------------------------------ sparring_partner
class TradeOff(BaseModel):
    option: str
    pros: str
    cons: str
    complexity: str = Field(description="time/memory complexity, e.g. 'O(B·T²·D) memory'")


class ChangeRequest(BaseModel):
    """partial_executor가 수행할 국소 수정 명세."""

    summary: str
    target_symbols: List[str] = Field(description="'Class', 'Class.method' or 'function' in the codebase")
    instructions: str
    expected_shape_changes: List[str] = Field(description="e.g. 'attn.out: (B,T,D) -> (B,T,D) unchanged'")


class SparringTurn(BaseModel):
    reply: str = Field(description="Korean reply to the user")
    options: List[TradeOff]
    recommendation: str
    questions: List[str]
    proposed_change: Optional[ChangeRequest] = Field(
        description="null while still discussing; set once a concrete code change is agreed"
    )
    escalate: bool = Field(
        description="true when the request needs a new architecture / substantial rewrite → hand off to the planner"
    )


# --------------------------------------------------------------- planner
class DimBinding(BaseModel):
    symbol: str = Field(description="single symbol, e.g. 'B', 'T', 'D'")
    value: int = Field(description="small value used for validation runs")
    meaning: str


class TensorSpec(BaseModel):
    name: str
    shape: List[str] = Field(description="symbols or arithmetic over symbols, e.g. ['B', 'T', 'D*4']")
    dtype: Literal["float32", "float16", "bfloat16", "int64", "bool"] = "float32"


class ModuleSpec(BaseModel):
    path: str = Field(description="attribute path from the entry module, e.g. 'encoder.layers.0.attn'")
    role: str
    input_shape: List[str]
    output_shape: List[str]
    complexity: str


class ModelSpec(BaseModel):
    name: str
    summary: str
    dims: List[DimBinding]
    inputs: List[TensorSpec] = Field(description="positional forward() arguments, in order")
    outputs: List[TensorSpec]
    modules: List[ModuleSpec]
    entry_class: str = Field(description="nn.Module class in model.py instantiated by the validator")
    init_kwargs_json: str = Field(description="JSON object of constructor kwargs using small validation values")
    causal_axis: Optional[str] = Field(
        description="symbol of the time axis if outputs at t must not depend on inputs after t; else null"
    )
    train_batch_size: int
    shape_diagram: str = Field(description="ASCII forward-pass diagram annotated with shapes")
    design_decisions: List[str]
    open_questions: List[str]

    def dim_values(self) -> Dict[str, int]:
        return {d.symbol: d.value for d in self.dims}

    def init_kwargs(self) -> dict:
        return json.loads(self.init_kwargs_json or "{}")


# -------------------------------------------------- executor / partial
class CodeFile(BaseModel):
    path: str = Field(description="relative path, e.g. 'model.py'")
    content: str


class CodeBundle(BaseModel):
    files: List[CodeFile]
    entry_file: str = Field(description="file that defines ModelSpec.entry_class")
    notes: str


class SymbolPatch(BaseModel):
    file: str
    symbol: str = Field(description="'Class', 'Class.method' or 'function'; created if missing")
    new_source: str = Field(description="complete replacement source of that symbol only")


class PatchSet(BaseModel):
    patches: List[SymbolPatch]
    new_files: List[CodeFile]
    notes: str


# ------------------------------------------------------------- validator
class ShapeRecord(BaseModel):
    module: str
    output_shape: str
    expected: Optional[str] = None
    ok: Optional[bool] = None


class LeakageProbe(BaseModel):
    checked: bool
    leaked: bool = False
    max_abs_diff: float = 0.0
    detail: str = ""


class LightningProbe(BaseModel):
    """Trainer(fast_dev_run=1)로 LightningModule을 1 step 학습/검증한 결과."""

    checked: bool
    file: Optional[str] = None
    lit_class: Optional[str] = None
    constructed_by: Optional[str] = None     # lit(model) | lit(**init_kwargs) | lit()
    batch_format: Optional[str] = None       # regression | classification (통과한 target 관례)
    loss: Optional[float] = None
    params_updated: Optional[bool] = None
    validation_step: Optional[str] = None    # ran | absent
    error: Optional[str] = None


class MetaDetail(BaseModel):
    """meta pass가 실패한 원인. 판정에는 쓰지 않지만 '왜 실측으로 폴백했는지'를 사람/reflector가 볼 수 있게."""

    kind: str                                # data_dependent | unsupported_op | shape_error | error
    stage: str                               # build | forward
    op: Optional[str] = None                 # nonzero, item, _unique2 ...
    module: Optional[str] = None             # 실패 시점에 forward 중이던 가장 안쪽 서브모듈
    location: Optional[str] = None           # 생성 코드의 file:line + 그 줄
    error: str = ""


class ValidationReport(BaseModel):
    status: Literal["pass", "fail", "skipped"]
    failed_stage: Optional[str] = None      # compile | import | build | forward | backward | shape | numerical | leakage | lightning | tests
    error: Optional[str] = None
    traceback: Optional[str] = None
    meta_pass: Optional[str] = None         # ok | data_dependent | unsupported_op | shape_error | error
    meta_detail: Optional[MetaDetail] = None
    shape_log: List[ShapeRecord] = []
    shape_mismatches: List[str] = []
    param_count: int = 0
    param_mb: float = 0.0
    activation_mb: float = 0.0
    est_train_activation_mb: float = 0.0
    est_adam_state_mb: float = 0.0
    nan_detected: bool = False
    unused_params: List[str] = []
    leakage: LeakageProbe = LeakageProbe(checked=False)
    lightning: LightningProbe = LightningProbe(checked=False)
    # 생성된 test_*.py를 pytest로 실행한 결과. None = 테스트 파일 없음
    tests_status: Optional[Literal["pass", "fail", "not_run"]] = None
    tests_detail: str = ""
    duration_s: float = 0.0
    stderr_tail: str = ""


# ------------------------------------------------------------- reflector
class Risk(BaseModel):
    category: Literal["shape", "oom", "leakage", "numerical", "gradient", "performance", "other"]
    severity: Literal["low", "medium", "high"]
    evidence: str = Field(description="quote from the validation log or code")
    fix: str


class Reflection(BaseModel):
    diagnosis: str = Field(description="Korean root-cause analysis")
    risks: List[Risk]
    action: Literal["patch", "accept", "ask_user"]
    change: Optional[ChangeRequest] = Field(description="required when action == 'patch'")


# ---------------------------------------------------------- human review
class ReviewDecision(BaseModel):
    action: Literal["approve", "revise", "reject", "escalate"]
    feedback: str = ""
