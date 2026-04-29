"""Four-direction anti-drift tests between schema.json and Pydantic models.

Direction 1: Pydantic -> JSON Schema field sets must match
Direction 2: JSON Schema examples -> Pydantic parsing must succeed
Direction 3: Pydantic model_dump -> jsonschema.validate must pass
Direction 4: Enum values in schema.json == Literal values in Pydantic
"""
import json

import jsonschema

from pet_schema.models import (
    ActionDistribution,
    ActionLabel,
    BowlInfo,
    EarPositionLabel,
    PetFeederEvent,
    PostureLabel,
    SceneInfo,
)


class TestDirection1PydanticToSchemaFields:
    def test_top_level_required_fields(self, v1_schema):
        pydantic_schema = PetFeederEvent.model_json_schema()
        schema_required = set(v1_schema["required"])
        pydantic_required = set(pydantic_schema.get("required", []))
        assert schema_required == pydantic_required, (
            f"Required mismatch: schema={schema_required}, pydantic={pydantic_required}"
        )

    def test_top_level_property_names(self, v1_schema):
        pydantic_schema = PetFeederEvent.model_json_schema()
        schema_props = set(v1_schema["properties"].keys())
        pydantic_props = set(pydantic_schema.get("properties", {}).keys())
        assert schema_props == pydantic_props, (
            f"Property mismatch: schema={schema_props}, pydantic={pydantic_props}"
        )


class TestDirection2SchemaExamplesToPydantic:
    def test_all_examples_parse(self, v1_examples):
        for i, example in enumerate(v1_examples):
            event = PetFeederEvent.model_validate(example["output"])
            assert event.schema_version == "1.0", f"Example {i} parse failed"


class TestDirection3PydanticDumpToJsonschema:
    def test_valid_event_roundtrip(self, v1_schema, valid_eating_output):
        event = PetFeederEvent.model_validate(valid_eating_output)
        dumped = json.loads(event.model_dump_json())
        jsonschema.validate(dumped, v1_schema)

    def test_no_pet_roundtrip(self, v1_schema, valid_no_pet_output):
        event = PetFeederEvent.model_validate(valid_no_pet_output)
        dumped = json.loads(event.model_dump_json())
        jsonschema.validate(dumped, v1_schema)


class TestDirection4EnumSync:
    def _get_pet_schema(self, v1_schema) -> dict:
        for option in v1_schema["properties"]["pet"]["oneOf"]:
            if option.get("type") == "object":
                return option
        raise AssertionError("No object type in pet oneOf")

    def test_species_enum(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_enum = set(pet_schema["properties"]["species"]["enum"])
        pydantic_enum = {"cat", "dog", "unknown"}
        assert schema_enum == pydantic_enum

    def test_action_primary_enum(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_enum = set(pet_schema["properties"]["action"]["properties"]["primary"]["enum"])
        pydantic_enum = set(ActionLabel.__args__)
        assert schema_enum == pydantic_enum

    def test_action_distribution_keys(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_keys = set(
            pet_schema["properties"]["action"]["properties"]["distribution"]["required"]
        )
        pydantic_keys = set(ActionDistribution.model_fields.keys())
        assert schema_keys == pydantic_keys

    def test_posture_enum(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_enum = set(
            pet_schema["properties"]["body_signals"]["properties"]["posture"]["enum"]
        )
        pydantic_enum = set(PostureLabel.__args__)
        assert schema_enum == pydantic_enum

    def test_ear_position_enum(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_enum = set(
            pet_schema["properties"]["body_signals"]["properties"]["ear_position"]["enum"]
        )
        pydantic_enum = set(EarPositionLabel.__args__)
        assert schema_enum == pydantic_enum

    def test_food_type_enum(self, v1_schema):
        # v3.4.0: field is Optional — JSON schema uses oneOf [null, string-enum]
        import typing

        schema_enum = set(
            v1_schema["properties"]["bowl"]["properties"]["food_type_visible"][
                "oneOf"
            ][1]["enum"]
        )
        annotation = BowlInfo.model_fields["food_type_visible"].annotation
        # annotation is `Literal[...] | None`; pull the Literal arm
        args = typing.get_args(annotation)
        literal_args = [a for a in args if a is not type(None)]
        assert len(literal_args) == 1
        pydantic_enum = set(typing.get_args(literal_args[0]))
        assert schema_enum == pydantic_enum

    def test_lighting_enum(self, v1_schema):
        schema_enum = set(
            v1_schema["properties"]["scene"]["properties"]["lighting"]["enum"]
        )
        annotation = SceneInfo.model_fields["lighting"].annotation
        pydantic_enum = set(annotation.__args__)
        assert schema_enum == pydantic_enum

    def test_image_quality_enum(self, v1_schema):
        schema_enum = set(
            v1_schema["properties"]["scene"]["properties"]["image_quality"]["enum"]
        )
        annotation = SceneInfo.model_fields["image_quality"].annotation
        pydantic_enum = set(annotation.__args__)
        assert schema_enum == pydantic_enum
