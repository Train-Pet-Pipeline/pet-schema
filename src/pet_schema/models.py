"""Pydantic v2 models for PetFeederEvent schema v1.0."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class ActionDistribution(BaseModel):
    """Action probability distribution. All 6 values must sum to 1.0 +/- 0.01."""

    eating: float
    drinking: float
    sniffing_only: float
    leaving_bowl: float
    sitting_idle: float
    other: float

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
    primary: ActionLabel
    distribution: ActionDistribution


class SpeedDistribution(BaseModel):
    """Eating speed distribution. All 3 values must sum to 1.0 +/- 0.01."""

    fast: float
    normal: float
    slow: float

    @model_validator(mode="after")
    def check_sum(self) -> SpeedDistribution:
        total = self.fast + self.normal + self.slow
        if abs(total - 1.0) > 0.01:
            msg = f"speed distribution sum is {total:.4f}, must be 1.0 +/- 0.01"
            raise ValueError(msg)
        return self


class EatingMetrics(BaseModel):
    speed: SpeedDistribution
    engagement: float
    abandoned_midway: float


class Mood(BaseModel):
    alertness: float
    anxiety: float
    engagement: float


PostureLabel = Literal["relaxed", "tense", "hunched", "unobservable"]
EarPositionLabel = Literal["forward", "flat", "rotating", "unobservable"]


class BodySignals(BaseModel):
    posture: PostureLabel
    ear_position: EarPositionLabel


class AnomalySignals(BaseModel):
    vomit_gesture: float
    food_rejection: float
    excessive_sniffing: float
    lethargy: float
    aggression: float


class PetInfo(BaseModel):
    species: Literal["cat", "dog", "unknown"]
    breed_estimate: str
    id_tag: str
    id_confidence: float
    action: ActionInfo
    eating_metrics: EatingMetrics
    mood: Mood
    body_signals: BodySignals
    anomaly_signals: AnomalySignals


class BowlInfo(BaseModel):
    food_fill_ratio: float | None = None
    water_fill_ratio: float | None = None
    food_type_visible: Literal["dry", "wet", "mixed", "unknown"]


class SceneInfo(BaseModel):
    lighting: Literal["bright", "dim", "infrared_night"]
    image_quality: Literal["clear", "blurry", "partially_occluded"]
    confidence_overall: float


class PetFeederEvent(BaseModel):
    schema_version: Literal["1.0"]
    pet_present: bool
    pet_count: int
    pet: PetInfo | None = None
    bowl: BowlInfo
    scene: SceneInfo
    narrative: str
