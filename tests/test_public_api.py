"""Guard the pet-schema 2.0 public API so downstream imports stay stable."""


def test_all_exports_importable_from_top_level():
    import pet_schema

    for name in pet_schema.__all__:
        assert hasattr(pet_schema, name), f"__all__ lists {name!r} but it's missing"


def test_petfeederevent_stays_top_level_importable_for_downstream():
    # pet-annotation / pet-train / pet-data (still on v1) import this path
    from pet_schema import PetFeederEvent  # noqa: F401


def test_phase_1_contracts_importable():
    from pet_schema import (  # noqa: F401
        Annotation,
        EdgeFormat,
        EvaluationReport,
        ExperimentRecipe,
        ModelCard,
        Sample,
        TrainerConfig,
    )


def test_schema_version_is_semver():
    import re

    from pet_schema import SCHEMA_VERSION

    assert re.match(r"^\d+\.\d+\.\d+$", SCHEMA_VERSION), (
        f"SCHEMA_VERSION must be SemVer X.Y.Z, got {SCHEMA_VERSION!r}"
    )
