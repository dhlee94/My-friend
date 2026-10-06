import pytest

from my_friend.tools.arxiv import parse_feed
from my_friend.tools.code_patch import PatchError, extract_symbol, outline, replace_symbol
from my_friend.tools.sandbox import SubprocessSandbox, resolve_dim

from .fixtures import CAUSAL_FORWARD, LEAKY_FORWARD, model_source, tiny_spec

pytest.importorskip("torch")


# ------------------------------------------------------------- code_patch
def test_replace_method_keeps_class_indent():
    src = model_source(LEAKY_FORWARD)
    out = replace_symbol(src, "TinyTS.forward", CAUSAL_FORWARD)
    assert "left pad only" in out and "centered" not in out
    assert "    def forward(self, x):" in out
    assert extract_symbol(out, "TinyTS.__init__") == extract_symbol(src, "TinyTS.__init__")


def test_add_new_method_and_function():
    src = model_source(CAUSAL_FORWARD)
    out = replace_symbol(src, "TinyTS.extra", "def extra(self):\n    return 1\n")
    out = replace_symbol(out, "helper", "def helper(x):\n    return x\n")
    names = "\n".join(outline(out))
    assert "TinyTS.extra" in names and "def helper" in names


def test_decorated_symbol_replaced_entirely():
    src = "import functools\n\n@functools.lru_cache()\ndef f(x):\n    return x\n"
    out = replace_symbol(src, "f", "def f(x):\n    return 2 * x\n")
    assert "lru_cache" not in out and "2 * x" in out


def test_bad_patch_rejected():
    with pytest.raises(PatchError):
        replace_symbol(model_source(CAUSAL_FORWARD), "TinyTS.forward", "def forward(self:\n  pass")
    with pytest.raises(PatchError):
        replace_symbol("x = 1\n", "Missing.method", "def method(self): pass")


# ---------------------------------------------------------------- sandbox
def test_resolve_dim():
    dims = {"B": 2, "T": 16, "D": 8}
    assert resolve_dim("T+2", dims) == 18
    assert resolve_dim("D*4", dims) == 32
    assert resolve_dim("T//4", dims) == 4
    with pytest.raises(ValueError):
        resolve_dim("__import__('os')", dims)
    with pytest.raises(ValueError):
        resolve_dim("Z", dims)


@pytest.fixture(scope="module")
def sandbox():
    return SubprocessSandbox(timeout_s=120)


def test_harness_pass_with_shape_log(sandbox):
    v = sandbox.validate({"model.py": model_source(CAUSAL_FORWARD)}, tiny_spec(), "model.py")
    assert v.status == "pass", v.error
    assert v.meta_pass == "ok"
    log = {r.module: r for r in v.shape_log}
    assert log["conv"].output_shape == "2x8x16" and log["conv"].ok
    assert v.leakage.checked and not v.leakage.leaked
    assert v.param_count > 0 and v.est_train_activation_mb > v.activation_mb


def test_harness_detects_future_leakage(sandbox):
    v = sandbox.validate({"model.py": model_source(LEAKY_FORWARD)}, tiny_spec(), "model.py")
    # centered pad k=3: 길이 보존 → shape은 통과, 누수만 잡혀야 한다
    assert v.status == "fail" and v.failed_stage == "leakage"
    assert v.leakage.leaked and v.leakage.max_abs_diff > 1e-5


def test_harness_shape_mismatch_and_runtime_error(sandbox):
    bad_spec = tiny_spec(init_kwargs_json='{"d_in": 4, "d_model": 8, "kernel": 3}')  # 입력 F=3과 불일치
    v = sandbox.validate({"model.py": model_source(CAUSAL_FORWARD)}, bad_spec, "model.py")
    assert v.status == "fail" and v.failed_stage == "forward"
    assert "mat1 and mat2" in v.error

    spec = tiny_spec()
    spec.modules[1].output_shape = ["B", "T", "D"]   # 명세 쪽이 틀린 경우도 mismatch로 드러나야 함
    v = sandbox.validate({"model.py": model_source(CAUSAL_FORWARD)}, spec, "model.py")
    assert v.failed_stage == "shape" and "conv: expected 2x16x8, got 2x8x16" in v.error


def test_harness_compile_error_and_no_spec(sandbox):
    v = sandbox.validate({"model.py": "def f(:\n"}, tiny_spec(), "model.py")
    assert v.failed_stage == "compile"
    v = sandbox.validate({"model.py": model_source(CAUSAL_FORWARD)}, None, None)
    assert v.status == "skipped"


def test_harness_meta_fallback_for_data_dependent_ops(sandbox):
    fwd = CAUSAL_FORWARD.replace(
        "return self.head(h.transpose(1, 2))        # (B, T, 1)",
        "y = self.head(h.transpose(1, 2))\n    _ = torch.nonzero(y > 0)\n    return y",
    )
    v = sandbox.validate({"model.py": model_source(fwd)}, tiny_spec(), "model.py")
    assert v.status == "pass", v.error
    assert v.meta_pass != "ok"


GOOD_TEST = """import torch
from model import TinyTS


def test_output_shape():
    torch.manual_seed(0)
    m = TinyTS(3, 8, 3)
    assert m(torch.randn(2, 16, 3)).shape == (2, 16, 1)
"""


def test_generated_tests_run_and_pass(sandbox):
    files = {"model.py": model_source(CAUSAL_FORWARD), "test_model.py": GOOD_TEST}
    v = sandbox.validate(files, tiny_spec(), "model.py")
    assert v.status == "pass", v.error
    assert v.tests_status == "pass" and "1 passed" in v.tests_detail


def test_failing_generated_test_fails_validation(sandbox):
    files = {"model.py": model_source(CAUSAL_FORWARD),
             "test_model.py": GOOD_TEST.replace("(2, 16, 1)", "(2, 16, 2)")}
    v = sandbox.validate(files, tiny_spec(), "model.py")
    assert v.status == "fail" and v.failed_stage == "tests" and v.tests_status == "fail"
    assert "test_output_shape" in v.error and "assert" in v.tests_detail
    assert v.leakage.checked and v.param_count > 0          # 하네스 결과는 보존


def test_broken_test_file_is_a_failure(sandbox):
    files = {"model.py": model_source(CAUSAL_FORWARD), "test_model.py": "from model import Missing\n"}
    v = sandbox.validate(files, tiny_spec(), "model.py")
    assert v.failed_stage == "tests" and v.tests_status == "fail"


def test_tests_not_run_when_harness_fails_but_run_without_spec(sandbox):
    files = {"model.py": model_source(LEAKY_FORWARD), "test_model.py": GOOD_TEST}
    v = sandbox.validate(files, tiny_spec(), "model.py")
    assert v.failed_stage == "leakage" and v.tests_status is None
    v = sandbox.validate({"model.py": model_source(CAUSAL_FORWARD), "test_model.py": GOOD_TEST}, None, None)
    assert v.status == "skipped" and v.tests_status == "pass"


def test_sandbox_rejects_path_escape(sandbox):
    v = sandbox.validate({"../evil.py": "x=1"}, None, None)
    assert v.status == "fail" and "escapes" in v.error


# ------------------------------------------------------------------ arxiv
def test_parse_feed():
    xml = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/2401.00001v2</id><title>Mamba
      Things</title><summary> SSM  stuff </summary><published>2024-01-01T00:00:00Z</published>
      <author><name>A</name></author><category term="cs.LG"/></entry></feed>"""
    (e,) = parse_feed(xml)
    assert e.arxiv_id == "2401.00001v2" and e.title == "Mamba Things" and e.categories == ["cs.LG"]
