import importlib.metadata

from pet_schema.version import SCHEMA_VERSION


def test_schema_version_matches_pyproject():
    """SCHEMA_VERSION must always match the installed package version in pyproject.toml."""
    installed = importlib.metadata.version("pet-schema")
    assert SCHEMA_VERSION == installed, (
        f"SCHEMA_VERSION={SCHEMA_VERSION!r} is out of sync with "
        f"pyproject.toml installed version {installed!r}"
    )
