from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from pet_schema.enums import Modality

_VALIDATED_BY_PATTERN = re.compile(r"^(github-actions|operator):[A-Za-z0-9_.\-]+$")


class QuantConfig(BaseModel):
    """Quantization configuration for a model checkpoint."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["gptq", "awq", "ptq_int8", "qat", "fp16", "none"]
    bits: int | None = None
    group_size: int | None = None
    calibration_dataset_uri: str | None = None


class EdgeArtifact(BaseModel):
    """Edge-deployment artifact metadata for a compiled model file."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["rkllm", "rknn", "onnx", "gguf"]
    target_hardware: list[str]
    artifact_uri: str
    sha256: str
    size_bytes: int
    min_firmware: str | None = None
    input_shape: dict[str, list[int]]


class ResourceSpec(BaseModel):
    """Hardware resource requirements for a training run."""

    model_config = ConfigDict(extra="forbid")

    gpu_count: int
    gpu_memory_gb: int
    cpu_count: int
    estimated_hours: float


class HardwareValidation(BaseModel):
    """Result of a manual real-hardware validation run, written back to ModelCard.

    Provenance is encoded in ``validated_by``:

    - ``github-actions:<workflow_run_id>`` when written by automation
    - ``operator:<github_username>``       when written by a human release manager
    """

    model_config = ConfigDict(extra="forbid")

    device_id: str
    firmware_version: str
    validated_at: datetime
    latency_ms_p50: float
    latency_ms_p95: float
    accuracy: float | None = None
    kl_divergence: float | None = None
    validated_by: str
    notes: str | None = None

    @field_validator("validated_by")
    @classmethod
    def _check_validated_by(cls, v: str) -> str:
        """Enforce provenance prefix so audit tooling can parse the source."""
        if not _VALIDATED_BY_PATTERN.fullmatch(v):
            raise ValueError(
                r"validated_by must match ^(github-actions|operator):[A-Za-z0-9_.\-]+$"
            )
        return v


class DeploymentStatus(BaseModel):
    """OTA backend deployment result, written back to ModelCard.deployment_history."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    state: Literal["pending", "deployed", "rolled_back", "failed"]
    deployed_at: datetime
    manifest_uri: str | None = None
    error: str | None = None
    notes: str | None = None


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
    recipe_id: str | None = None
    hydra_config_sha: str
    git_shas: dict[str, str]
    dataset_versions: dict[str, str]

    # Artifact
    checkpoint_uri: str

    # Optional downstream
    quantization: QuantConfig | None = None
    edge_artifacts: list[EdgeArtifact] = []
    intermediate_artifacts: dict[str, str] = {}
    deployment_history: list[DeploymentStatus] = []

    # Lineage
    parent_models: list[str] = []
    lineage_role: Literal["teacher", "student", "sft_base", "dpo_output", "fused"] | None = None

    # Metrics
    metrics: dict[str, float]
    gate_status: Literal["pending", "passed", "failed"]

    # Tracing
    trained_at: datetime
    trained_by: str
    clearml_task_id: str | None = None
    dvc_exp_sha: str | None = None
    resolved_config_uri: str | None = None
    notes: str | None = None

    # Hardware gate result (written after real-device validation)
    hardware_validation: HardwareValidation | None = None

    def to_manifest_entry(self) -> dict[str, object]:
        """Serialize for pet-ota manifest.json — JSON-ready, keeps Nones."""
        return self.model_dump(mode="json", exclude_none=False)
