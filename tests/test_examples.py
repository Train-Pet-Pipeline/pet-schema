"""Tests that all few-shot examples pass schema validation and Pydantic parsing."""
import json

from pet_schema import validate_output
from pet_schema.models import PetFeederEvent


class TestFewShotExamples:
    def test_all_examples_pass_validator(self, v1_examples):
        for i, example in enumerate(v1_examples):
            output_str = json.dumps(example["output"])
            result = validate_output(output_str)
            assert result.valid, (
                f"Example {i} ({example['scene_desc']}) failed validation: {result.errors}"
            )

    def test_all_examples_pass_pydantic(self, v1_examples):
        for i, example in enumerate(v1_examples):
            event = PetFeederEvent.model_validate(example["output"])
            assert event.schema_version == "1.0", f"Example {i} has wrong schema_version"

    def test_no_pet_example_has_null_pet(self, v1_examples):
        for example in v1_examples:
            output = example["output"]
            if not output["pet_present"]:
                assert output["pet"] is None, (
                    f"Example '{example['scene_desc']}': pet_present=false but pet is not null"
                )

    def test_pet_example_has_pet(self, v1_examples):
        for example in v1_examples:
            output = example["output"]
            if output["pet_present"]:
                assert output["pet"] is not None, (
                    f"Example '{example['scene_desc']}': pet_present=true but pet is null"
                )
