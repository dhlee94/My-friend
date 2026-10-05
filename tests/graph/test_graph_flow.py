"""E2E 그래프 흐름: 시나리오 LLM + 실제 subprocess 샌드박스(torch) + MemorySaver."""
import pytest

from my_friend.agents.base import AgentDeps
from my_friend.agents.schemas import (
    ChangeRequest,
    CodeBundle,
    CodeFile,
    PaperSummary,
    PatchSet,
    Reflection,
    Risk,
    RouteDecision,
    ScoutReport,
    ScoutSearchPlan,
    SparringTurn,
    SymbolPatch,
    TradeOff,
)
from my_friend.graph.builder import build_graph
from my_friend.graph.routing import route_after_reflection, route_after_validation
from my_friend.graph.session import Session
from my_friend.tools.arxiv import ArxivEntry
from my_friend.tools.sandbox import SubprocessSandbox

from .fixtures import CAUSAL_FORWARD, LEAKY_FORWARD, ScriptedLLM, model_source, tiny_spec

pytest.importorskip("torch")


class FakeArxiv:
    def __init__(self):
        self.queries = []

    def search(self, query, max_results=10):
        self.queries.append(query)
        return [ArxivEntry("2401.00001v1", "Causal Conv Forecasting", "abstract", "2024-01-01", ["A"], ["cs.LG"])]


def make(script, **deps_kw):
    llm = ScriptedLLM(script)
    deps = AgentDeps(llm=llm, sandbox=SubprocessSandbox(timeout_s=120), **deps_kw)
    return llm, Session(build_graph(deps))


CHANGE = ChangeRequest(
    summary="centered pad → left pad",
    target_symbols=["TinyTS.forward"],
    instructions="make the conv causal",
    expected_shape_changes=["conv: (B,D,T) unchanged"],
)
PATCH = PatchSet(patches=[SymbolPatch(file="model.py", symbol="TinyTS.forward", new_source=CAUSAL_FORWARD)],
                 new_files=[], notes="causal padding")


def sparring_turn(change=None, reply="좌측 패딩으로 바꾸면 미래 누수가 사라집니다."):
    return SparringTurn(
        reply=reply,
        options=[TradeOff(option="left pad", pros="causal", cons="none", complexity="O(BTD²k)")],
        recommendation="left pad",
        questions=["dilation도 쓸까요?"],
        proposed_change=change,
    )


def scout_steps():
    return {
        ScoutSearchPlan: [ScoutSearchPlan(arxiv_query='abs:"causal convolution"', local_query="causal conv")],
        ScoutReport: [ScoutReport(
            papers=[PaperSummary(arxiv_id="2401.00001v1", title="Causal Conv Forecasting", published="2024-01-01",
                                 key_idea="k", relevance="r"),
                    PaperSummary(arxiv_id="9999.99999", title="Hallucinated", published="2024", key_idea="k", relevance="r")],
            trends=["causal conv"], digest="요약")],
    }


# ------------------------------------------------------------------ full build
def test_full_build_plan_revise_approve_build_validate_reflect():
    spec_v1 = tiny_spec(name="TinyTS-v1")
    spec_v2 = tiny_spec(name="TinyTS-v2")
    llm, s = make({
        RouteDecision: [RouteDecision(mode="full_build", reason="new model")],
        type(spec_v1): [spec_v1, spec_v2],
        CodeBundle: [CodeBundle(files=[CodeFile(path="model.py", content=model_source(CAUSAL_FORWARD)),
                                       CodeFile(path="lit_module.py", content="import torch\n")],
                                entry_file="model.py", notes="built")],
        Reflection: [Reflection(diagnosis="문제 없음", risks=[], action="accept", change=None)],
    })

    r = s.send("시계열 예측용 causal conv 모델을 새로 설계해줘")
    assert r.pending_review["kind"] == "plan" and "TinyTS-v1" in r.pending_review["body"]
    assert set(r.pending_review["options"]) == {"approve", "revise", "reject"}

    r = s.resume("revise", "dilation 없이 가자")
    assert r.pending_review["kind"] == "plan" and "TinyTS-v2" in r.pending_review["body"]
    planner_prompt = [u for name, u, _ in llm.calls if name == "ModelSpec"][-1]
    assert "dilation 없이 가자" in planner_prompt and "TinyTS-v1" in planner_prompt  # 피드백 + 이전 명세

    r = s.resume("approve")
    assert r.done
    st = r.state
    assert st["validation"].status == "pass"
    assert st["reflection"].action == "accept"
    assert set(st["files"]) == {"model.py", "lit_module.py"}
    assert [m["name"] for m in r.new_messages] == ["executor", "validator", "reflector"]
    assert llm.remaining() == {}
    # effort가 노드별로 전달되는지
    assert ("CodeBundle", "high") in {(n, e) for n, _, e in llm.calls}


def test_full_build_reflector_patches_leak_then_accepts():
    llm, s = make({
        RouteDecision: [RouteDecision(mode="full_build", reason="new")],
        type(tiny_spec()): [tiny_spec()],
        CodeBundle: [CodeBundle(files=[CodeFile(path="model.py", content=model_source(LEAKY_FORWARD))],
                                entry_file="model.py", notes="built")],
        Reflection: [
            Reflection(diagnosis="미래 누수", action="patch", change=CHANGE,
                       risks=[Risk(category="leakage", severity="high", evidence="max|Δ|>0", fix="left pad")]),
            Reflection(diagnosis="해결됨", risks=[], action="accept", change=None),
        ],
        PatchSet: [PATCH],
    })
    s.send("모델 만들어줘")
    r = s.resume("approve")
    assert r.done
    names = [m["name"] for m in r.new_messages]
    assert names.count("validator") == 2 and names.index("partial_executor") > names.index("reflector")
    assert r.state["validation"].status == "pass"
    assert "left pad only" in r.state["files"]["model.py"]
    assert r.state["iteration"] == 1
    # partial executor에는 전체 파일이 아니라 대상 심볼 원문 + outline이 전달
    pe_prompt = next(u for n, u, _ in llm.calls if n == "PatchSet")
    assert '<symbol name="TinyTS.forward" file="model.py">' in pe_prompt and "centered" in pe_prompt


def test_reflector_iteration_budget_stops_loop():
    bad_patch = PatchSet(patches=[SymbolPatch(file="model.py", symbol="TinyTS.forward", new_source=LEAKY_FORWARD)],
                         new_files=[], notes="no-op")
    leak = Reflection(diagnosis="누수", action="patch", change=CHANGE,
                      risks=[Risk(category="leakage", severity="high", evidence="e", fix="f")])
    llm, s = make({
        RouteDecision: [RouteDecision(mode="full_build", reason="new")],
        type(tiny_spec()): [tiny_spec()],
        CodeBundle: [CodeBundle(files=[CodeFile(path="model.py", content=model_source(LEAKY_FORWARD))],
                                entry_file="model.py", notes="")],
        Reflection: [leak] * 3,
        PatchSet: [bad_patch] * 2,
    })
    s.send("만들어줘")
    s.graph.update_state(s.config, {"max_iterations": 2})   # 승인 대기 중 예산 조정
    r = s.resume("approve")
    assert r.done and r.state["validation"].status == "fail"
    assert [m["name"] for m in r.new_messages].count("partial_executor") == 2
    assert llm.remaining() == {}


# -------------------------------------------------------------- sparring mode
def test_sparring_tikitaka_then_partial_update_on_previous_build():
    """턴1: full build(누수 있는 코드, reflector accept) → 턴2: sparring으로 국소 수정.
    MemorySaver가 턴1의 files/spec을 보존해야 턴2 partial update + forward 검증이 가능."""
    arxiv = FakeArxiv()
    llm, s = make({
        RouteDecision: [RouteDecision(mode="full_build", reason="new"), RouteDecision(mode="sparring", reason="local fix")],
        type(tiny_spec()): [tiny_spec()],
        CodeBundle: [CodeBundle(files=[CodeFile(path="model.py", content=model_source(LEAKY_FORWARD))],
                                entry_file="model.py", notes="")],
        Reflection: [Reflection(diagnosis="사용자 판단 필요", risks=[], action="ask_user", change=None)],
        **scout_steps(),
        SparringTurn: [sparring_turn(change=None, reply="패딩 방식부터 정하죠."), sparring_turn(change=CHANGE)],
        PatchSet: [PATCH],
    }, arxiv=arxiv)

    s.send("모델 만들어줘")
    r = s.resume("approve")
    assert r.pending_review["kind"] == "reflection"        # ask_user → interrupt
    r = s.resume("reject")
    assert r.done and r.state["validation"].failed_stage == "leakage"

    # ---- 턴2: sparring
    r = s.send("conv 패딩 때문에 누수 나는 것 같은데 어떻게 고칠까?")
    assert r.state["mode"] == "sparring"
    assert r.state["iteration"] == 0 and r.state["reflection"] is None     # 턴 단위 초기화
    assert r.pending_review["kind"] == "proposal" and "escalate" in r.pending_review["options"]
    assert [p.arxiv_id for p in r.state["scout"].papers] == ["2401.00001v1"]  # 환각 id 제거
    assert arxiv.queries == ['abs:"causal convolution"']

    r = s.resume("approve")                                 # 아직 change 없음 → 승인해도 아무 것도 안 함
    assert r.done and r.state["validation"] is None

    # 티키타카: revise로 대화 이어가기 (scout는 재실행 안 함)
    llm.queues[RouteDecision].append(RouteDecision(mode="sparring", reason="cont"))
    llm.queues[SparringTurn].appendleft(sparring_turn(change=None))
    llm.queues[ScoutSearchPlan].append(scout_steps()[ScoutSearchPlan][0])
    llm.queues[ScoutReport].append(scout_steps()[ScoutReport][0])
    r = s.send("다시 논의하자")
    r = s.resume("revise", "left pad로 가자")
    assert r.pending_review["kind"] == "proposal"
    sparring_prompt = [u for n, u, _ in llm.calls if n == "SparringTurn"][-1]
    assert "left pad로 가자" in sparring_prompt
    assert "class TinyTS" in sparring_prompt                # 이전 턴 코드가 세션에 남아 있음

    r = s.resume("approve")
    assert r.done
    assert r.state["validation"].status == "pass"           # 이전 턴 spec으로 forward + leakage 재검증
    assert "left pad only" in r.state["files"]["model.py"]
    assert [m["name"] for m in r.new_messages][-2:] == ["partial_executor", "validator"]
    assert llm.remaining() == {}


def test_sparring_escalate_to_full_build():
    llm, s = make({
        RouteDecision: [RouteDecision(mode="sparring", reason="q")],
        **scout_steps(),
        SparringTurn: [sparring_turn()],
        type(tiny_spec()): [tiny_spec()],
    }, arxiv=FakeArxiv())
    s.send("새 아키텍처가 필요할까?")
    r = s.resume("escalate", "그냥 새로 설계하자")
    assert r.pending_review["kind"] == "plan" and r.state["mode"] == "full_build"


def test_explicit_mode_skips_router_llm():
    llm, s = make({**scout_steps(), SparringTurn: [sparring_turn()]}, arxiv=FakeArxiv())
    r = s.send("질문", mode="sparring")
    assert r.state["route_reason"] == "user override"
    assert "RouteDecision" not in [n for n, _, _ in llm.calls]


def test_send_while_review_pending_raises():
    llm, s = make({**scout_steps(), SparringTurn: [sparring_turn()]}, arxiv=FakeArxiv())
    s.send("질문", mode="sparring")
    with pytest.raises(RuntimeError):
        s.send("또 질문")


# ----------------------------------------------------------------- routing
def test_routing_tables():
    from my_friend.agents.schemas import ValidationReport

    ok, bad = ValidationReport(status="pass"), ValidationReport(status="fail")
    assert route_after_validation({"mode": "full_build", "validation": ok}) == "reflector"
    assert route_after_validation({"mode": "sparring", "validation": ok}) == "__end__"
    assert route_after_validation({"mode": "sparring", "validation": bad}) == "reflector"
    patch = Reflection(diagnosis="", risks=[], action="patch", change=CHANGE)
    assert route_after_reflection({"reflection": patch, "iteration": 3, "max_iterations": 3}) == "partial_executor"
    assert route_after_reflection({"reflection": patch, "iteration": 4, "max_iterations": 3}) == "__end__"
