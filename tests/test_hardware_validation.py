from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pet_schema.model_card import HardwareValidation, ModelCard


def _base_card_kwargs() -> dict:
    return dict(
        id="recipe1_train_abc12345",
        version="1.0.0",
        modality="vision",
        task="classification",
        arch="qwen2_vl",
        training_recipe="vlm_sft_baseline",
        hydra_config_sha="sha123",
        git_shas={},
        dataset_versions={},
        checkpoint_uri="local:///ckpts/1",
        metrics={"accuracy": 0.9},
        gate_status="passed",
        trained_at=datetime(2026, 4, 20),
        trained_by="tester",
    )


def test_validated_by_github_actions_format_accepted():
    hv = HardwareValidation(
        device_id="rk3576-dev-01",
        firmware_version="1.2.3",
        validated_at=datetime.now(UTC),
        latency_ms_p50=42.0,
        latency_ms_p95=88.0,
        validated_by="github-actions:1234567890",
    )
    assert hv.validated_by.startswith("github-actions:")


def test_validated_by_operator_format_accepted():
    hv = HardwareValidation(
        device_id="rk3576-dev-01",
        firmware_version="1.2.3",
        validated_at=datetime.now(UTC),
        latency_ms_p50=42.0,
        latency_ms_p95=88.0,
        validated_by="operator:alice-smith.2",
    )
    assert hv.validated_by.startswith("operator:")


def test_validated_by_invalid_prefix_rejected():
    with pytest.raises(ValidationError):
        HardwareValidation(
            device_id="rk3576-dev-01",
            firmware_version="1.2.3",
            validated_at=datetime.now(UTC),
            latency_ms_p50=42.0,
            latency_ms_p95=88.0,
            validated_by="random-person",
        )


def test_validated_by_empty_suffix_rejected():
    with pytest.raises(ValidationError):
        HardwareValidation(
            device_id="rk3576-dev-01",
            firmware_version="1.2.3",
            validated_at=datetime.now(UTC),
            latency_ms_p50=42.0,
            latency_ms_p95=88.0,
            validated_by="operator:",
        )


def test_model_card_hardware_validation_defaults_none():
    card = ModelCard(**_base_card_kwargs())
    assert card.hardware_validation is None


def test_model_card_hardware_validation_roundtrip():
    hv = HardwareValidation(
        device_id="rk3576-dev-01",
        firmware_version="1.2.3",
        validated_at=datetime.now(UTC),
        latency_ms_p50=42.0,
        latency_ms_p95=88.0,
        accuracy=0.91,
        kl_divergence=0.05,
        validated_by="operator:alice",
        notes="smoke-only run",
    )
    card = ModelCard(**_base_card_kwargs(), hardware_validation=hv)
    dumped = card.model_dump(mode="json")
    assert dumped["hardware_validation"]["validated_by"] == "operator:alice"
    reloaded = ModelCard.model_validate(dumped)
    assert reloaded.hardware_validation is not None
    assert reloaded.hardware_validation.latency_ms_p95 == 88.0


def test_forward_compat_schema_21_card_loads_under_22():
    """A ModelCard without hardware_validation (schema 2.1.0 shape) loads fine."""
    old_card_json = {
        **_base_card_kwargs(),
        "trained_at": "2026-04-20T00:00:00",
    }
    card = ModelCard.model_validate(old_card_json)
    assert card.hardware_validation is None
