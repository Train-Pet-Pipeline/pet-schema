from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from pet_schema.enums import Modality


class QuantConfig(BaseModel):
    """Quantization configuration for a model checkpoint."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["gptq", "awq", "ptq_int8", "qat", "fp16", "none"]
    bits: Optional[int] = None
    group_size: Optional[int] = None
    calibration_dataset_uri: Optional[str] = None


class EdgeArtifact(BaseModel):
    """Edge-deployment artifact metadata for a compiled model file."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["rkllm", "rknn", "onnx", "gguf"]
    target_hardware: list[str]
    artifact_uri: str
    sha256: str
    size_bytes: int
    min_firmware: Optional[str] = None
    input_shape: dict[str, list[int]]


class ResourceSpec(BaseModel):
    """Hardware resource requirements for a training run."""

    model_config = ConfigDict(extra="forbid")

    gpu_count: int
    gpu_memory_gb: int
    cpu_count: int
    estimated_hours: float


class ModelCard(BaseModel):
    """Canonical model card contract shared across the Train-Pet-Pipeline."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    id: str
    version: str
    modality: Modality
    task: str
    arch: str

    # Reproducibility
    training_recipe: str
    recipe_id: Optional[str] = None
    hydra_config_sha: str
    git_shas: dict[str, str]
    dataset_versions: dict[str, str]

    # Artifact
    checkpoint_uri: str

    # Optional downstream
    quantization: Optional[QuantConfig] = None
    edge_artifact: Optional[EdgeArtifact] = None

    # Lineage
    parent_models: list[str] = []
    lineage_role: Optional[
        Literal["teacher", "student", "sft_base", "dpo_output", "fused"]
    ] = None

    # Metrics
    metrics: dict[str, float]
    gate_status: Literal["pending", "passed", "failed"]

    # Tracing
    trained_at: datetime
    trained_by: str
    clearml_task_id: Optional[str] = None
    dvc_exp_sha: Optional[str] = None
    notes: Optional[str] = None

    def to_manifest_entry(self) -> dict:
        """Serialize for pet-ota manifest.json — JSON-ready, keeps Nones."""
        return self.model_dump(mode="json", exclude_none=False)
