"""E2E 그래프 흐름: 시나리오 LLM + 실제 subprocess 샌드박스(torch) + MemorySaver / SqliteSaver."""
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
from my_friend.graph.builder import build_graph, make_checkpointer
from my_friend.graph.routing import route_after_reflection, route_after_sparring, route_by_mode
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


def make(script, checkpointer=None, thread_id=None, **deps_kw):
    llm = ScriptedLLM(script)
    deps = AgentDeps(llm=llm, sandbox=SubprocessSandbox(timeout_s=120), **deps_kw)
    return llm, Session(build_graph(deps, checkpointer), thread_id=thread_id)


CHANGE = ChangeRequest(
    summary="centered pad → left pad",
    target_symbols=["TinyTS.forward"],
    instructions="make the conv causal",
    expected_shape_changes=["conv: (B,D,T) unchanged"],
)
PATCH = PatchSet(patches=[SymbolPatch(file="model.py", symbol="TinyTS.forward", new_source=CAUSAL_FORWARD)],
                 new_files=[], notes="causal padding")


def sparring_turn(change=None, reply="좌측 패딩으로 바꾸면 미래 누수가 사라집니다.", escalate=False):
    return SparringTurn(
        reply=reply,
        options=[TradeOff(option="left pad", pros="causal", cons="none", complexity="O(BTD²k)")],
        recommendation="left pad",
        questions=["dilation도 쓸까요?"],
        proposed_change=change,
        escalate=escalate,
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


def scout_plans(n=1):
    """arxiv/retriever 없이 scout가 돌 때: 검색 계획 LLM 호출만 일어나고 요약(ScoutReport)은 생략된다."""
    return {ScoutSearchPlan: [ScoutSearchPlan(arxiv_query="q", local_query="q")] * n}


# ------------------------------------------------------------------ full build
def test_full_build_plan_revise_approve_build_validate_reflect():
    spec_v1 = tiny_spec(name="TinyTS-v1")
    spec_v2 = tiny_spec(name="TinyTS-v2")
    llm, s = make({
        **scout_plans(),
        RouteDecision: [RouteDecision(mode="full_build", reason="new model")],
        type(spec_v1): [spec_v1, spec_v2],
        CodeBundle: [CodeBundle(files=[CodeFile(path="model.py", content=model_source(CAUSAL_FORWARD)),
                                       CodeFile(path="lit_module.py", content="import torch\n")],
                                entry_file="model.py", notes="built")],
        Reflection: [Reflection(diagnosis="문제 없음", risks=[], action="accept", change=None)],
    })

    r = s.send("시계열 예측용 causal conv 모델을 새로 설계해줘")
    assert r.pending_review["kind"] == "plan" and "TinyTS-v1" in r.pending_review["body"]
    # 시작은 무조건 scout → router → planner
    assert [m["name"] for m in r.new_messages] == ["user", "paper_scout", "planner"]
    assert [n for n, _, _ in llm.calls] == ["ScoutSearchPlan", "RouteDecision", "ModelSpec"]
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
        **scout_plans(),
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


def test_executor_keeps_only_python_and_runs_generated_tests():
    test_src = ("import torch\nfrom model import TinyTS\n\n\ndef test_shape():\n"
                "    assert TinyTS(3, 8, 3)(torch.randn(2, 16, 3)).shape == (2, 16, 1)\n")
    llm, s = make({
        **scout_plans(),
        RouteDecision: [RouteDecision(mode="full_build", reason="new")],
        type(tiny_spec()): [tiny_spec()],
        CodeBundle: [CodeBundle(files=[CodeFile(path="model.py", content=model_source(CAUSAL_FORWARD)),
                                       CodeFile(path="test_model.py", content=test_src),
                                       CodeFile(path="README.md", content="# docs"),
                                       CodeFile(path="notes.md", content="notes")],
                                entry_file="model.py", notes="built")],
        Reflection: [Reflection(diagnosis="ok", risks=[], action="accept", change=None)],
    })
    s.send("모델 만들어줘")
    r = s.resume("approve")
    assert set(r.state["files"]) == {"model.py", "test_model.py"}
    executor_msg = next(m["content"] for m in r.new_messages if m["name"] == "executor")
    assert "README.md" in executor_msg and "제외" in executor_msg
    assert r.state["validation"].tests_status == "pass"
    assert "generated tests: pass" in next(m["content"] for m in r.new_messages if m["name"] == "validator")


def test_reflector_iteration_budget_stops_loop():
    bad_patch = PatchSet(patches=[SymbolPatch(file="model.py", symbol="TinyTS.forward", new_source=LEAKY_FORWARD)],
                         new_files=[], notes="no-op")
    leak = Reflection(diagnosis="누수", action="patch", change=CHANGE,
                      risks=[Risk(category="leakage", severity="high", evidence="e", fix="f")])
    llm, s = make({
        **scout_plans(),
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
        Reflection: [Reflection(diagnosis="사용자 판단 필요", risks=[], action="ask_user", change=None),
                     Reflection(diagnosis="누수 해결", risks=[], action="accept", change=None)],
        ScoutSearchPlan: [scout_steps()[ScoutSearchPlan][0]] * 2,
        ScoutReport: [scout_steps()[ScoutReport][0]] * 2,
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
    assert arxiv.queries == ['abs:"causal convolution"'] * 2           # 턴마다 scout 선행
    router_prompt = [u for n, u, _ in llm.calls if n == "RouteDecision"][-1]
    assert "<scout_digest>" in router_prompt                              # router가 scout 결과를 본다

    r = s.resume("approve")                                 # 아직 change 없음 → 승인해도 아무 것도 안 함
    assert r.done and r.state["validation"] is None

    # 티키타카: revise로 대화 이어가기 (revise 루프에서는 scout 재실행 안 함)
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
    assert r.state["reflection"].action == "accept"         # 수렴점: sparring 트랙도 reflector 최종 검수
    assert "left pad only" in r.state["files"]["model.py"]
    assert [m["name"] for m in r.new_messages][-3:] == ["partial_executor", "validator", "reflector"]
    assert llm.remaining() == {}


def test_sparring_auto_escalates_to_planner():
    """에스컬레이션 엣지: sparring_partner가 escalate=True면 사용자 승인 없이 planner로 직행."""
    llm, s = make({
        RouteDecision: [RouteDecision(mode="sparring", reason="q")],
        **scout_steps(),
        SparringTurn: [sparring_turn(change=CHANGE, reply="국소 수정으로는 안 됩니다.", escalate=True)],
        type(tiny_spec()): [tiny_spec()],
    }, arxiv=FakeArxiv())
    r = s.send("attention을 SSM으로 통째로 바꾸면?")
    assert r.pending_review["kind"] == "plan"
    assert r.state["mode"] == "full_build" and r.state["change"] is None
    assert [m["name"] for m in r.new_messages] == ["user", "paper_scout", "sparring_partner", "planner"]
    planner_prompt = next(u for n, u, _ in llm.calls if n == "ModelSpec")
    assert "<scout_digest>" in planner_prompt and "국소 수정으로는 안 됩니다." in planner_prompt


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


def test_sqlite_checkpointer_survives_restart(tmp_path):
    """프로세스 재시작 = 새 그래프 + 새 커넥션. 같은 thread_id면 승인 대기 상태부터 이어간다."""
    db = tmp_path / "ckpt.sqlite"
    script = {
        **scout_plans(),
        RouteDecision: [RouteDecision(mode="full_build", reason="new")],
        type(tiny_spec()): [tiny_spec()],
    }
    _, s1 = make(script, checkpointer=make_checkpointer(db))
    s1.send("모델 만들어줘")
    assert s1.pending_review()["kind"] == "plan"

    llm2, s2 = make({
        CodeBundle: [CodeBundle(files=[CodeFile(path="model.py", content=model_source(CAUSAL_FORWARD))],
                                entry_file="model.py", notes="")],
        Reflection: [Reflection(diagnosis="ok", risks=[], action="accept", change=None)],
    }, checkpointer=make_checkpointer(db), thread_id=s1.thread_id)
    assert s2.pending_review()["kind"] == "plan"
    r = s2.resume("approve")
    assert r.done and r.state["validation"].status == "pass"
    assert r.state["spec"].name == "TinyTS"                 # pydantic 상태가 sqlite 왕복 후 복원
    assert llm2.remaining() == {}


def test_per_node_llm_routing():
    """llms에 지정한 노드만 해당 LLM으로, 나머지는 기본 llm으로 간다."""
    router_llm = ScriptedLLM({RouteDecision: [RouteDecision(mode="full_build", reason="new")]})
    llm = ScriptedLLM({**scout_plans(), type(tiny_spec()): [tiny_spec()]})
    deps = AgentDeps(llm=llm, llms={"router": router_llm}, sandbox=SubprocessSandbox(timeout_s=120))
    r = Session(build_graph(deps)).send("모델 만들어줘")
    assert r.pending_review["kind"] == "plan"
    assert [n for n, _, _ in router_llm.calls] == ["RouteDecision"]
    assert [n for n, _, _ in llm.calls] == ["ScoutSearchPlan", "ModelSpec"]


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
    assert route_by_mode({"mode": "sparring"}) == "sparring_partner"
    assert route_by_mode({"mode": "full_build"}) == "planner"
    assert route_after_sparring({"sparring": sparring_turn(escalate=True)}) == "planner"
    assert route_after_sparring({"sparring": sparring_turn()}) == "human_review"
    patch = Reflection(diagnosis="", risks=[], action="patch", change=CHANGE)
    assert route_after_reflection({"reflection": patch, "iteration": 3, "max_iterations": 3}) == "partial_executor"
    assert route_after_reflection({"reflection": patch, "iteration": 4, "max_iterations": 3}) == "__end__"
