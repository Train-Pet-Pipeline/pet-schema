"""Tests for VLM output validator."""
import copy
import json

from pet_schema.validator import validate_output


class TestValidateOutputValid:
    def test_valid_eating_event(self, valid_eating_output):
        result = validate_output(json.dumps(valid_eating_output))
        assert result.valid is True
        assert result.errors == []

    def test_valid_no_pet_event(self, valid_no_pet_output):
        result = validate_output(json.dumps(valid_no_pet_output))
        assert result.valid is True
        assert result.errors == []


class TestValidateOutputInvalidJson:
    def test_invalid_json_string(self):
        result = validate_output("not json at all")
        assert result.valid is False
        assert any("JSON" in e for e in result.errors)

    def test_empty_string(self):
        result = validate_output("")
        assert result.valid is False


class TestValidateOutputSchemaErrors:
    def test_missing_required_field(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        del data["narrative"]
        result = validate_output(json.dumps(data))
        assert result.valid is False

    def test_invalid_enum_value(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["body_signals"] = {"posture": "sleeping", "ear_position": "forward"}
        result = validate_output(json.dumps(data))
        assert result.valid is False


class TestValidateOutputExtraValidations:
    def test_pet_present_true_but_pet_null(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"] = None
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("pet_present" in e for e in result.errors)

    def test_pet_present_false_but_pet_not_null(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet_present"] = False
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("pet_present" in e for e in result.errors)

    def test_distribution_sum_too_high(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["action"]["distribution"] = {
            "eating": 0.80, "drinking": 0.10,
            "sniffing_only": 0.10, "leaving_bowl": 0.10,
            "sitting_idle": 0.00, "other": 0.00,
        }
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("distribution" in e for e in result.errors)

    def test_speed_sum_too_high(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["eating_metrics"]["speed"] = {"fast": 0.50, "normal": 0.50, "slow": 0.50}
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("speed" in e for e in result.errors)

    def test_primary_not_max_in_distribution(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["action"]["primary"] = "drinking"
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("primary" in e for e in result.errors)

    def test_narrative_too_long(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["narrative"] = "这" * 81
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("narrative" in e for e in result.errors)

    def test_narrative_exactly_80(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["narrative"] = "这" * 80
        result = validate_output(json.dumps(data))
        assert result.valid is True

    def test_speed_all_zero_when_not_eating(self, valid_eating_output):
        """speed 分布全 0 在非进食场景下应该合法。"""
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["action"]["primary"] = "other"
        data["pet"]["action"]["distribution"] = {
            "eating": 0.00, "drinking": 0.00,
            "sniffing_only": 0.00, "leaving_bowl": 0.00,
            "sitting_idle": 0.00, "other": 1.00,
        }
        data["pet"]["eating_metrics"]["speed"] = {"fast": 0.00, "normal": 0.00, "slow": 0.00}
        result = validate_output(json.dumps(data))
        assert result.valid is True
        assert not any("speed" in e for e in result.errors)

    def test_speed_nonzero_must_sum_to_one(self, valid_eating_output):
        """speed 分布非 0 但求和不为 1 应报错。"""
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["action"]["primary"] = "other"
        data["pet"]["action"]["distribution"] = {
            "eating": 0.00, "drinking": 0.00,
            "sniffing_only": 0.00, "leaving_bowl": 0.00,
            "sitting_idle": 0.00, "other": 1.00,
        }
        data["pet"]["eating_metrics"]["speed"] = {"fast": 0.30, "normal": 0.30, "slow": 0.00}
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("speed" in e for e in result.errors)
