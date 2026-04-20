from pet_schema.version import SCHEMA_VERSION


def test_schema_version_is_semver_2_0_0():
    assert SCHEMA_VERSION == "2.0.0"
