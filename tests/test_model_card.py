from datetime import datetime

import pytest
from pydantic import ValidationError

from pet_schema.model_card import EdgeArtifact, ModelCard, QuantConfig, ResourceSpec


def _mc(**overrides) -> ModelCard:
    base = dict(
        id="recipe1_train_abc12345",
        version="1.0.0",
        modality="vision",
        task="classification",
        arch="qwen2_vl",
        training_recipe="vlm_sft_baseline",
        recipe_id="vlm_sft_baseline",
        hydra_config_sha="sha123",
        git_shas={},
        dataset_versions={},
        checkpoint_uri="local:///ckpts/1",
        quantization=None,
        parent_models=[],
        lineage_role=None,
        metrics={"accuracy": 0.9},
        gate_status="passed",
        trained_at=datetime(2026, 4, 20),
        trained_by="tester",
        clearml_task_id=None,
        dvc_exp_sha=None,
        notes=None,
    )
    base.update(overrides)
    return ModelCard(**base)


def test_quant_config_requires_method_literal():
    q = QuantConfig(method="gptq", bits=4, group_size=128, calibration_dataset_uri=None)
    assert q.method == "gptq"


def test_quant_config_rejects_unknown_method():
    with pytest.raises(ValidationError):
        QuantConfig(method="magic", bits=None, group_size=None, calibration_dataset_uri=None)


def test_edge_artifact_format_validated():
    ea = EdgeArtifact(
        format="rkllm",
        target_hardware=["rk3576"],
        artifact_uri="local:///edge/a.rkllm",
        sha256="abc",
        size_bytes=1234,
        min_firmware=None,
        input_shape={"input": [1, 3, 224, 224]},
    )
    assert ea.format == "rkllm"


def test_edge_artifact_rejects_unknown_format():
    with pytest.raises(ValidationError):
        EdgeArtifact(
            format="coreml",
            target_hardware=["rk3576"],
            artifact_uri="local:///x",
            sha256="abc",
            size_bytes=1,
            min_firmware=None,
            input_shape={"x": [1]},
        )


def test_resource_spec_requires_all_fields():
    rs = ResourceSpec(gpu_count=1, gpu_memory_gb=24, cpu_count=8, estimated_hours=2.5)
    assert rs.gpu_count == 1
    with pytest.raises(ValidationError):
        ResourceSpec(gpu_count=1, gpu_memory_gb=24, cpu_count=8)  # type: ignore[call-arg]


def test_model_card_to_manifest_entry_has_required_keys():
    ea = EdgeArtifact(
        format="rkllm",
        target_hardware=["rk3576"],
        artifact_uri="local:///edge/a.rkllm",
        sha256="abc",
        size_bytes=1234,
        min_firmware=None,
        input_shape={"input": [1, 3, 224, 224]},
    )
    mc = _mc(edge_artifacts=[ea])
    entry = mc.to_manifest_entry()
    assert entry["id"] == mc.id
    assert entry["version"] == mc.version
    assert entry["checkpoint_uri"] == mc.checkpoint_uri
    assert entry["edge_artifacts"][0]["format"] == "rkllm"


def test_model_card_rejects_extra_field():
    with pytest.raises(ValidationError):
        _mc(surprise="field")


def test_model_card_parent_models_defaults_sensibly():
    mc = _mc()
    assert mc.parent_models == []


def test_model_card_lineage_role_literal_values():
    for role in ["teacher", "student", "sft_base", "dpo_output", "fused"]:
        mc = _mc(lineage_role=role)
        assert mc.lineage_role == role
    mc_none = _mc(lineage_role=None)
    assert mc_none.lineage_role is None


def test_model_card_lineage_role_rejects_unknown():
    with pytest.raises(ValidationError):
        _mc(lineage_role="other")


def test_model_card_gate_status_literal():
    with pytest.raises(ValidationError):
        _mc(gate_status="skipped")


def test_model_card_to_manifest_entry_excludes_none_false():
    # When optional fields are None they should still appear (exclude_none=False)
    mc = _mc()
    entry = mc.to_manifest_entry()
    assert "quantization" in entry
    assert entry["quantization"] is None
