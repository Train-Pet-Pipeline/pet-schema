"""Annotation contracts — discriminator 按 annotator_type（非 modality）。

Spec: docs/superpowers/specs/2026-04-21-phase-2-debt-repayment-design.md §2
"""
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator

from pet_schema.enums import Modality


class BaseAnnotation(BaseModel):
    """Base class for all annotation paradigms."""

    annotation_id: str
    target_id: str
    annotator_type: Literal["llm", "classifier", "rule", "human"]
    annotator_id: str
    modality: Modality
    schema_version: str
    created_at: datetime
    storage_uri: str | None = None


class LLMAnnotation(BaseAnnotation):
    """Annotation produced by an LLM/VLM; stores raw response and parsed structured output."""

    annotator_type: Literal["llm"] = "llm"
    prompt_hash: str
    raw_response: str
    parsed_output: dict[str, Any]


class ClassifierAnnotation(BaseAnnotation):
    """Annotation produced by a classifier model (e.g. audio CNN)."""

    annotator_type: Literal["classifier"] = "classifier"
    predicted_class: str
    class_probs: dict[str, float]
    logits: list[float] | None = None


class RuleAnnotation(BaseAnnotation):
    """Annotation produced by a deterministic rule."""

    annotator_type: Literal["rule"] = "rule"
    rule_id: str
    rule_output: dict[str, Any]


class HumanAnnotation(BaseAnnotation):
    """Annotation produced by a human reviewer."""

    annotator_type: Literal["human"] = "human"
    reviewer: str
    decision: str
    notes: str | None = None


Annotation = Annotated[
    LLMAnnotation | ClassifierAnnotation | RuleAnnotation | HumanAnnotation,
    Discriminator("annotator_type"),
]


class DpoPair(BaseModel):
    """A chosen/rejected annotation pair for DPO training."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    chosen_annotation_id: str
    rejected_annotation_id: str
    target_id: str
    modality: Modality
    preference_source: Literal["human", "rule", "auto"]
    reason: str | None = None
    created_at: datetime
    schema_version: str
