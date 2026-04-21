from pet_schema.version import SCHEMA_VERSION


def test_schema_version_is_semver_2_3_1():
    assert SCHEMA_VERSION == "2.3.1"
