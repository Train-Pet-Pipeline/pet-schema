from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricResult(BaseModel):
    """A single evaluation metric result."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    higher_is_better: bool


class GateCheck(BaseModel):
    """A quality gate check comparing an actual metric value against a threshold."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    threshold: float
    comparator: Literal["ge", "le", "eq"]
    passed: bool
    actual_value: float

    @classmethod
    def evaluate(
        cls,
        metric_name: str,
        actual: float,
        threshold: float,
        comparator: str,
    ) -> GateCheck:
        """Evaluate a gate check and return a GateCheck instance with the result.

        Args:
            metric_name: Name of the metric being checked.
            actual: The actual observed value.
            threshold: The threshold to compare against.
            comparator: One of "ge", "le", "eq".

        Returns:
            GateCheck instance with passed field set based on comparison.

        Raises:
            ValueError: If comparator is not one of "ge", "le", "eq".
        """
        if comparator == "ge":
            passed = actual >= threshold
        elif comparator == "le":
            passed = actual <= threshold
        elif comparator == "eq":
            passed = math.isclose(actual, threshold)
        else:
            raise ValueError(f"unknown comparator: {comparator!r}")
        return cls(
            metric_name=metric_name,
            threshold=threshold,
            comparator=comparator,  # type: ignore[arg-type]
            passed=passed,
            actual_value=actual,
        )


class EvaluationReport(BaseModel):
    """A complete evaluation report for a model, content-addressed by report_id."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(pattern=r"^[a-f0-9]{16}$")
    model_card_id: str
    evaluator_type: str
    dataset_uri: str
    metrics: list[MetricResult]
    gate_checks: list[GateCheck]
    gate_status: Literal["passed", "failed", "pending"]
    artifacts: dict[str, str]
    evaluated_at: datetime
    clearml_task_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _auto_compute_report_id(cls, data: Any) -> Any:
        """Auto-compute report_id from content when not supplied."""
        if not isinstance(data, dict):
            return data
        if data.get("report_id"):
            return data
        required = ("model_card_id", "evaluator_type", "dataset_uri", "evaluated_at")
        if not all(data.get(k) is not None for k in required):
            return data  # let downstream validation surface the missing-field error
        ts = data["evaluated_at"]
        if isinstance(ts, datetime):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)
        payload = (
            f"{data['model_card_id']}|{data['evaluator_type']}|"
            f"{data['dataset_uri']}|{ts_str}"
        )
        data["report_id"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return data
