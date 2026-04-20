import pytest
from pydantic import ValidationError

from pet_schema.configs import (
    ConverterConfig,
    DatasetConfig,
    EvaluatorConfig,
    ResourcesSection,
    TrainerConfig,
)
from pet_schema.metric import GateCheck
from pet_schema.recipe import ArtifactRef


def test_resources_section_defaults():
    r = ResourcesSection()
    assert r.gpu_count == 0
    assert r.gpu_memory_gb == 0
    assert r.cpu_count == 1
    assert r.estimated_hours == 1.0


def test_trainer_config_validates():
    t = TrainerConfig(
        type="pet_train.vlm_sft",
        args={"lora_r": 128, "batch_size": 4},
        resources=ResourcesSection(gpu_count=1, gpu_memory_gb=24),
    )
    assert t.type == "pet_train.vlm_sft"
    assert t.args["lora_r"] == 128
    assert t.resources.gpu_count == 1


def test_trainer_config_rejects_missing_type():
    with pytest.raises(ValidationError):
        TrainerConfig(
            args={},
            resources=ResourcesSection(),
        )  # type: ignore[call-arg]


def test_trainer_config_does_not_dive_into_args():
    # Opaque dict — plugin-specific. Anything dict-shaped is accepted.
    t = TrainerConfig(
        type="x",
        args={"nested": {"deep": [1, 2, 3]}, "any": True},
        resources=ResourcesSection(),
    )
    assert t.args["nested"]["deep"] == [1, 2, 3]


def test_evaluator_config_gates_default_empty():
    e = EvaluatorConfig(type="pet_eval.cls", args={})
    assert e.gates == []


def test_evaluator_config_accepts_gatecheck_list():
    gc = GateCheck.evaluate("acc", 0.9, 0.8, "ge")
    e = EvaluatorConfig(type="pet_eval.cls", args={}, gates=[gc])
    assert len(e.gates) == 1
    assert e.gates[0].passed is True


def test_converter_config_calibration_optional():
    c = ConverterConfig(type="pet_quantize.rkllm", args={})
    assert c.calibration is None


def test_converter_config_accepts_artifact_ref():
    ref = ArtifactRef(ref_type="dataset", ref_value="calib_v1")
    c = ConverterConfig(type="pet_quantize.rkllm", args={}, calibration=ref)
    assert c.calibration is not None
    assert c.calibration.ref_value == "calib_v1"


def test_dataset_config_modality_literal():
    d = DatasetConfig(type="pet_data.vision_loader", args={}, modality="vision")
    assert d.modality == "vision"


def test_dataset_config_rejects_unknown_modality():
    with pytest.raises(ValidationError):
        DatasetConfig(type="x", args={}, modality="thermal")  # type: ignore[arg-type]


def test_configs_reject_extra_fields():
    with pytest.raises(ValidationError):
        TrainerConfig(
            type="x", args={}, resources=ResourcesSection(),
            surprise="field",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        DatasetConfig(
            type="x", args={}, modality="vision",
            surprise="field",  # type: ignore[call-arg]
        )
