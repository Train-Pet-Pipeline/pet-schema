"""Tests for Pydantic v2 models."""
import copy

import pytest
from pydantic import ValidationError

from pet_schema.models import PetFeederEvent


class TestPetFeederEventParsing:
    def test_parse_valid_eating(self, valid_eating_output):
        event = PetFeederEvent.model_validate(valid_eating_output)
        assert event.pet_present is True
        assert event.pet is not None
        assert event.pet.action.primary == "eating"
        assert event.pet.body_signals.posture == "relaxed"
        assert event.pet.mood.alertness == 0.28

    def test_parse_valid_no_pet(self, valid_no_pet_output):
        event = PetFeederEvent.model_validate(valid_no_pet_output)
        assert event.pet_present is False
        assert event.pet is None

    def test_action_distribution_sum_valid(self, valid_eating_output):
        event = PetFeederEvent.model_validate(valid_eating_output)
        dist = event.pet.action.distribution
        total = (
            dist.eating + dist.drinking + dist.sniffing_only
            + dist.leaving_bowl + dist.sitting_idle + dist.other
        )
        assert abs(total - 1.0) <= 0.01

    def test_action_distribution_sum_invalid(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["action"]["distribution"] = {
            "eating": 0.90, "drinking": 0.90,
            "sniffing_only": 0.00, "leaving_bowl": 0.00,
            "sitting_idle": 0.00, "other": 0.00,
        }
        with pytest.raises(ValidationError, match="distribution"):
            PetFeederEvent.model_validate(data)

    def test_speed_distribution_sum_invalid(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["eating_metrics"]["speed"] = {"fast": 0.90, "normal": 0.90, "slow": 0.00}
        with pytest.raises(ValidationError, match="speed"):
            PetFeederEvent.model_validate(data)

    def test_invalid_posture_enum(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["body_signals"] = {"posture": "sleeping", "ear_position": "forward"}
        with pytest.raises(ValidationError):
            PetFeederEvent.model_validate(data)

    def test_invalid_species(self, valid_eating_output):
        data = copy.deepcopy(valid_eating_output)
        data["pet"]["species"] = "hamster"
        with pytest.raises(ValidationError):
            PetFeederEvent.model_validate(data)
