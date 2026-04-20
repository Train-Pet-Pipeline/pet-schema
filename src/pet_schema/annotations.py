from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator

from pet_schema.enums import Modality
from pet_schema.models import PetFeederEvent


class BaseAnnotation(BaseModel):
    """Base class for all annotation types."""

    model_config = ConfigDict(extra="forbid")

    annotation_id: str
    sample_id: str
    annotator_type: Literal["vlm", "cnn", "human", "rule"]
    annotator_id: str
    modality: Modality
    created_at: datetime
    schema_version: str


class VisionAnnotation(BaseAnnotation):
    """Annotation produced by a vision model; wraps a parsed PetFeederEvent."""

    modality: Literal["vision"] = "vision"
    raw_response: str
    parsed: PetFeederEvent
    prompt_hash: str


class AudioAnnotation(BaseAnnotation):
    """Annotation produced by an audio/CNN model."""

    modality: Literal["audio"] = "audio"
    predicted_class: str
    class_probs: dict[str, float]
    logits: list[float] | None = None


Annotation = Annotated[
    VisionAnnotation | AudioAnnotation,
    Discriminator("modality"),
]


class DpoPair(BaseModel):
    """A chosen/rejected annotation pair for DPO training."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    chosen_annotation_id: str
    rejected_annotation_id: str
    preference_source: Literal["human", "rule", "auto"]
    reason: str | None = None
