from datetime import datetime

import pytest
from pydantic import ValidationError

from pet_schema.metric import EvaluationReport, GateCheck, MetricResult


# ---- MetricResult ----

def test_metric_result_instantiates():
    m = MetricResult(name="accuracy", value=0.91, higher_is_better=True)
    assert m.name == "accuracy"
    assert m.higher_is_better is True


def test_metric_result_rejects_extra():
    with pytest.raises(ValidationError):
        MetricResult(name="x", value=0.1, higher_is_better=True, surprise=1)  # type: ignore[call-arg]


# ---- GateCheck.evaluate ----

def test_gate_check_evaluate_ge_passes():
    g = GateCheck.evaluate("accuracy", 0.9, 0.85, "ge")
    assert g.passed is True
    assert g.actual_value == 0.9
    assert g.comparator == "ge"


def test_gate_check_evaluate_ge_fails():
    g = GateCheck.evaluate("accuracy", 0.8, 0.9, "ge")
    assert g.passed is False


def test_gate_check_evaluate_le_passes():
    g = GateCheck.evaluate("latency_ms", 100.0, 200.0, "le")
    assert g.passed is True


def test_gate_check_evaluate_le_fails():
    g = GateCheck.evaluate("latency_ms", 300.0, 200.0, "le")
    assert g.passed is False


def test_gate_check_evaluate_eq_passes_within_tolerance():
    g = GateCheck.evaluate("ratio", 1.0 + 1e-10, 1.0, "eq")
    assert g.passed is True


def test_gate_check_evaluate_eq_fails_beyond_tolerance():
    g = GateCheck.evaluate("ratio", 1.1, 1.0, "eq")
    assert g.passed is False


def test_gate_check_evaluate_rejects_unknown_comparator():
    with pytest.raises(ValueError):
        GateCheck.evaluate("x", 1.0, 1.0, "lt")


# ---- EvaluationReport ----

def _report(**overrides):
    base = dict(
        model_card_id="mc1",
        evaluator_type="pet_eval.vlm_cls",
        dataset_uri="local:///tmp/ds",
        metrics=[MetricResult(name="accuracy", value=0.9, higher_is_better=True)],
        gate_checks=[],
        gate_status="passed",
        artifacts={},
        evaluated_at=datetime(2026, 4, 20, 12, 0, 0),
        clearml_task_id=None,
    )
    base.update(overrides)
    return EvaluationReport(**base)


def test_evaluation_report_auto_computes_deterministic_report_id():
    r1 = _report()
    r2 = _report()
    assert r1.report_id == r2.report_id, "same args => same report_id"
    assert len(r1.report_id) == 16


def test_evaluation_report_report_id_hex_format():
    r = _report()
    import re
    assert re.fullmatch(r"[a-f0-9]{16}", r.report_id), r.report_id


def test_evaluation_report_rejects_invalid_report_id_regex():
    with pytest.raises(ValidationError):
        _report(report_id="not-hex-string!")


def test_evaluation_report_accepts_supplied_valid_report_id():
    r = _report(report_id="0123456789abcdef")
    assert r.report_id == "0123456789abcdef"


def test_evaluation_report_changes_when_input_changes():
    r1 = _report()
    r2 = _report(model_card_id="mc2")
    assert r1.report_id != r2.report_id


def test_gate_status_accepts_three_literals():
    for s in ["passed", "failed", "pending"]:
        r = _report(gate_status=s)
        assert r.gate_status == s


def test_gate_status_rejects_unknown():
    with pytest.raises(ValidationError):
        _report(gate_status="skipped")


def test_evaluation_report_with_failing_gate_check():
    gc = GateCheck.evaluate("accuracy", 0.5, 0.9, "ge")
    r = _report(gate_checks=[gc], gate_status="failed")
    assert r.gate_status == "failed"
    assert r.gate_checks[0].passed is False


def test_evaluation_report_with_pending_gate_check():
    gc = GateCheck(
        metric_name="accuracy",
        threshold=0.9,
        comparator="ge",
        passed=False,
        actual_value=0.0,
    )
    r = _report(gate_checks=[gc], gate_status="pending")
    assert r.gate_status == "pending"
