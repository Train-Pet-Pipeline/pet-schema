from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pet_schema.model_card import DeploymentStatus, ModelCard


def _base_card_kwargs():
    return dict(
        id="x",
        version="v1",
        modality="vision",
        task="t",
        arch="a",
        training_recipe="r",
        hydra_config_sha="s",
        git_shas={},
        dataset_versions={},
        checkpoint_uri="file:///tmp/x",
        metrics={},
        gate_status="pending",
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        trained_by="me",
    )


def test_deployment_status_valid():
    """DeploymentStatus round-trips with valid fields."""
    s = DeploymentStatus(
        backend="local",
        state="deployed",
        deployed_at=datetime(2026, 4, 21, tzinfo=UTC),
        manifest_uri="file:///tmp/m.json",
    )
    assert s.state == "deployed"


def test_deployment_status_state_literal_rejected():
    """Unknown state value raises ValidationError."""
    with pytest.raises(ValidationError):
        DeploymentStatus(
            backend="local",
            state="bogus",  # type: ignore[arg-type]
            deployed_at=datetime(2026, 4, 21, tzinfo=UTC),
        )


def test_modelcard_deployment_history_default_empty():
    """ModelCard.deployment_history defaults to empty list."""
    card = ModelCard(**_base_card_kwargs())
    assert card.deployment_history == []


def test_modelcard_deployment_history_append():
    """ModelCard accepts a populated deployment_history list."""
    s = DeploymentStatus(
        backend="local",
        state="deployed",
        deployed_at=datetime(2026, 4, 21, tzinfo=UTC),
    )
    card = ModelCard(**_base_card_kwargs(), deployment_history=[s])
    assert card.deployment_history[0].backend == "local"


def test_modelcard_edge_artifacts_default_empty():
    """ModelCard.edge_artifacts defaults to empty list."""
    card = ModelCard(**_base_card_kwargs())
    assert card.edge_artifacts == []


def test_modelcard_intermediate_artifacts_default_empty():
    """ModelCard.intermediate_artifacts defaults to empty dict."""
    card = ModelCard(**_base_card_kwargs())
    assert card.intermediate_artifacts == {}


def test_modelcard_extra_still_forbidden():
    """ModelCard still rejects unknown fields after schema additions."""
    with pytest.raises(ValidationError):
        ModelCard(**_base_card_kwargs(), bogus_field="x")
