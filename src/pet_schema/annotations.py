"""Annotation contracts — discriminator 按 annotator_type（非 modality）。

Spec: docs/superpowers/specs/2026-04-21-phase-2-debt-repayment-design.md §2
"""
from datetime import datetime
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Discriminator

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
    storage_uri: Optional[str] = None


class LLMAnnotation(BaseAnnotation):
    """Annotation produced by an LLM/VLM; stores raw response and parsed structured output."""

    annotator_type: Literal["llm"] = "llm"
    prompt_hash: str
    raw_response: str
    parsed_output: dict


class ClassifierAnnotation(BaseAnnotation):
    """Annotation produced by a classifier model (e.g. audio CNN)."""

    annotator_type: Literal["classifier"] = "classifier"
    predicted_class: str
    class_probs: dict[str, float]
    logits: Optional[list[float]] = None


class RuleAnnotation(BaseAnnotation):
    """Annotation produced by a deterministic rule."""

    annotator_type: Literal["rule"] = "rule"
    rule_id: str
    rule_output: dict


class HumanAnnotation(BaseAnnotation):
    """Annotation produced by a human reviewer."""

    annotator_type: Literal["human"] = "human"
    reviewer: str
    decision: str
    notes: Optional[str] = None


Annotation = Annotated[
    LLMAnnotation | ClassifierAnnotation | RuleAnnotation | HumanAnnotation,
    Discriminator("annotator_type"),
]


class DpoPair(BaseModel):
    """保留：由 human/llm annotation 衍生出的偏好对。"""

    pair_id: str
    chosen_annotation_id: str
    rejected_annotation_id: str
    target_id: str
    modality: Modality
    created_at: datetime
    schema_version: str
