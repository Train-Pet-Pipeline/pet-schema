from datetime import datetime

from pet_schema.adapters.manifest import build_manifest
from pet_schema.model_card import ModelCard
from pet_schema.version import SCHEMA_VERSION


def _mc(id_: str) -> ModelCard:
    return ModelCard(
        id=id_,
        version="1.0.0",
        modality="vision",
        task="classification",
        arch="q",
        training_recipe="r",
        recipe_id=None,
        hydra_config_sha="sha",
        git_shas={},
        dataset_versions={},
        checkpoint_uri="local:///c",
        quantization=None,
        parent_models=[],
        lineage_role=None,
        metrics={},
        gate_status="passed",
        trained_at=datetime(2026, 4, 20),
        trained_by="t",
        clearml_task_id=None,
        dvc_exp_sha=None,
        notes=None,
    )


def test_empty_list_yields_empty_models():
    m = build_manifest([])
    assert m["models"] == []
    assert m["schema_version"] == SCHEMA_VERSION
    assert "generated_at" in m


def test_order_preserved():
    cards = [_mc("a"), _mc("b")]
    m = build_manifest(cards)
    assert [e["id"] for e in m["models"]] == ["a", "b"]


def test_schema_version_matches_constant():
    from pet_schema.version import SCHEMA_VERSION
    m = build_manifest([])
    assert m["schema_version"] == SCHEMA_VERSION
