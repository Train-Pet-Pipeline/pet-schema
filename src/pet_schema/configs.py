from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from pet_schema.enums import Modality
from pet_schema.metric import GateCheck
from pet_schema.recipe import ArtifactRef


class ResourcesSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_count: int = 0
    gpu_memory_gb: int = 0
    cpu_count: int = 1
    estimated_hours: float = 1.0


class TrainerConfig(BaseModel):
    """Outer-shell Hydra target — plugin validates `args` internally."""

    model_config = ConfigDict(extra="forbid")

    type: str
    args: dict[str, Any]
    resources: ResourcesSection


class EvaluatorConfig(BaseModel):
    """Outer-shell Hydra target — plugin validates `args` internally."""

    model_config = ConfigDict(extra="forbid")

    type: str
    args: dict[str, Any]
    gates: list[GateCheck] = []


class ConverterConfig(BaseModel):
    """Outer-shell Hydra target — plugin validates `args` internally."""

    model_config = ConfigDict(extra="forbid")

    type: str
    args: dict[str, Any]
    calibration: Optional[ArtifactRef] = None


class DatasetConfig(BaseModel):
    """Outer-shell Hydra target — plugin validates `args` internally."""

    model_config = ConfigDict(extra="forbid")

    type: str
    args: dict[str, Any]
    modality: Modality
