"""Pydantic v2 models for PetFeederEvent schema v1.0."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionDistribution(BaseModel):
    """Action probability distribution. All 6 values must sum to 1.0 +/- 0.01."""

    model_config = ConfigDict(extra="forbid")

    eating: float = Field(ge=0.0, le=1.0)
    drinking: float = Field(ge=0.0, le=1.0)
    sniffing_only: float = Field(ge=0.0, le=1.0)
    leaving_bowl: float = Field(ge=0.0, le=1.0)
    sitting_idle: float = Field(ge=0.0, le=1.0)
    other: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self) -> ActionDistribution:
        total = (
            self.eating + self.drinking + self.sniffing_only
            + self.leaving_bowl + self.sitting_idle + self.other
        )
        if abs(total - 1.0) > 0.01:
            msg = f"action distribution sum is {total:.4f}, must be 1.0 +/- 0.01"
            raise ValueError(msg)
        return self


ActionLabel = Literal[
    "eating", "drinking", "sniffing_only", "leaving_bowl", "sitting_idle", "other"
]


class ActionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: ActionLabel
    distribution: ActionDistribution


class SpeedDistribution(BaseModel):
    """Eating speed distribution. All 3 values must sum to 1.0 +/- 0.01."""

    model_config = ConfigDict(extra="forbid")

    fast: float = Field(ge=0.0, le=1.0)
    normal: float = Field(ge=0.0, le=1.0)
    slow: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self) -> SpeedDistribution:
        total = self.fast + self.normal + self.slow
        if abs(total - 1.0) > 0.01:
            msg = f"speed distribution sum is {total:.4f}, must be 1.0 +/- 0.01"
            raise ValueError(msg)
        return self


class EatingMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speed: SpeedDistribution
    engagement: float = Field(ge=0.0, le=1.0)
    abandoned_midway: float = Field(ge=0.0, le=1.0)


class Mood(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alertness: float = Field(ge=0.0, le=1.0)
    anxiety: float = Field(ge=0.0, le=1.0)
    engagement: float = Field(ge=0.0, le=1.0)


PostureLabel = Literal["relaxed", "tense", "hunched", "unobservable"]
EarPositionLabel = Literal["forward", "flat", "rotating", "unobservable"]


class BodySignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posture: PostureLabel
    ear_position: EarPositionLabel


class AnomalySignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vomit_gesture: float = Field(ge=0.0, le=1.0)
    food_rejection: float = Field(ge=0.0, le=1.0)
    excessive_sniffing: float = Field(ge=0.0, le=1.0)
    lethargy: float = Field(ge=0.0, le=1.0)
    aggression: float = Field(ge=0.0, le=1.0)


class PetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    species: Literal["cat", "dog", "unknown"]
    breed_estimate: str = Field(min_length=1)
    id_tag: str = Field(min_length=1)
    id_confidence: float = Field(ge=0.0, le=1.0)
    action: ActionInfo
    eating_metrics: EatingMetrics
    mood: Mood
    body_signals: BodySignals
    anomaly_signals: AnomalySignals


class BowlInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    food_fill_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    water_fill_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    food_type_visible: Literal["dry", "wet", "mixed", "unknown"]


class SceneInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lighting: Literal["bright", "dim", "infrared_night"]
    image_quality: Literal["clear", "blurry", "partially_occluded"]
    confidence_overall: float = Field(ge=0.0, le=1.0)


class PetFeederEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    pet_present: bool
    pet_count: int = Field(ge=0, le=4)
    pet: PetInfo | None
    bowl: BowlInfo
    scene: SceneInfo
    narrative: str = Field(max_length=80)
