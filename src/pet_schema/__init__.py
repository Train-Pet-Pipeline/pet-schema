"""pet-schema public API.

Contracts for the multi-model pipeline: Sample / Annotation / ModelCard /
ExperimentRecipe / EvaluationReport / Hydra Structured Configs /
Training JSONL samples (SFTSample, DPOSample, ShareGPTTurn).
"""
from pet_schema.annotations import (
    Annotation,
    BaseAnnotation,
    ClassifierAnnotation,
    DpoPair,
    HumanAnnotation,
    LLMAnnotation,
    RuleAnnotation,
)
from pet_schema.configs import (
    ConverterConfig,
    DatasetConfig,
    EvaluatorConfig,
    ResourcesSection,
    TrainerConfig,
)
from pet_schema.enums import (
    BowlType,
    EdgeFormat,
    Lighting,
    Modality,
    PetSpecies,
    SourceType,
)
from pet_schema.metric import EvaluationReport, GateCheck, MetricResult
from pet_schema.model_card import (
    DeploymentStatus,
    EdgeArtifact,
    HardwareValidation,
    ModelCard,
    QuantConfig,
    ResourceSpec,
)
from pet_schema.models import PetFeederEvent  # legacy v1 — keep importable from top-level
from pet_schema.recipe import (
    AblationAxis,
    ArtifactRef,
    ExperimentRecipe,
    RecipeStage,
)
from pet_schema.renderer import render_prompt
from pet_schema.samples import (
    AudioSample,
    BaseSample,
    Sample,
    SensorSample,
    SourceInfo,
    VisionSample,
)
from pet_schema.training_samples import DPOSample, SFTSample, ShareGPTSFTSample, ShareGPTTurn
from pet_schema.validator import validate_output
from pet_schema.version import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "render_prompt",
    "validate_output",
    # legacy v1 model (downstream pin to this import path)
    "PetFeederEvent",
    # samples
    "BaseSample",
    "Sample",
    "VisionSample",
    "AudioSample",
    "SensorSample",
    "SourceInfo",
    # annotations
    "Annotation",
    "BaseAnnotation",
    "LLMAnnotation",
    "ClassifierAnnotation",
    "RuleAnnotation",
    "HumanAnnotation",
    "DpoPair",
    # model card
    "ModelCard",
    "QuantConfig",
    "EdgeArtifact",
    "ResourceSpec",
    "HardwareValidation",
    "DeploymentStatus",
    # recipe
    "ArtifactRef",
    "RecipeStage",
    "AblationAxis",
    "ExperimentRecipe",
    # metric
    "MetricResult",
    "GateCheck",
    "EvaluationReport",
    # configs
    "TrainerConfig",
    "EvaluatorConfig",
    "ConverterConfig",
    "DatasetConfig",
    "ResourcesSection",
    # enums
    "Modality",
    "EdgeFormat",
    "PetSpecies",
    "BowlType",
    "Lighting",
    "SourceType",
    # training samples (v3.2.0)
    "SFTSample",
    "DPOSample",
    "ShareGPTTurn",
    "ShareGPTSFTSample",
]
