"""v3.4.0 — bowl.food_type_visible accepts null when no bowl visible.

Schema changes from required Literal to Literal | None = None so that VLM
output for non-feeder scenes (no bowl in frame) passes validation.
"""
import pytest
from pydantic import ValidationError

from pet_schema.models import BowlInfo, PetFeederEvent


def test_bowlinfo_accepts_none_food_type():
    bowl = BowlInfo(
        food_fill_ratio=None, water_fill_ratio=None, food_type_visible=None
    )
    assert bowl.food_type_visible is None


def test_bowlinfo_accepts_omitted_food_type():
    bowl = BowlInfo(food_fill_ratio=None, water_fill_ratio=None)
    assert bowl.food_type_visible is None


def test_bowlinfo_still_rejects_unknown_string():
    with pytest.raises(ValidationError):
        BowlInfo(
            food_fill_ratio=None, water_fill_ratio=None, food_type_visible="kibble"
        )


def test_petfeedeerevent_no_pet_no_bowl():
    """Pet-absent scene with no bowl visible — common in non-feeder datasets."""
    event = PetFeederEvent(
        schema_version="1.0",
        pet_present=False,
        pet_count=0,
        pet=None,
        bowl=BowlInfo(
            food_fill_ratio=None, water_fill_ratio=None, food_type_visible=None
        ),
        scene={
            "lighting": "bright",
            "image_quality": "clear",
            "confidence_overall": 0.5,
        },
        narrative="empty room, no pets visible",
    )
    assert event.bowl.food_type_visible is None
