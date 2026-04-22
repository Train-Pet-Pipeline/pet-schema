"""Shared test fixtures for pet-schema tests."""
import json
from pathlib import Path

import pytest

VERSIONS_DIR = Path(__file__).parent.parent / "src" / "pet_schema" / "versions"


@pytest.fixture
def v1_examples() -> list[dict]:
    """Load v1.0 few-shot examples."""
    path = VERSIONS_DIR / "v1.0" / "few_shot_examples.json"
    return json.loads(path.read_text())


@pytest.fixture
def v1_schema() -> dict:
    """Load v1.0 JSON Schema."""
    path = VERSIONS_DIR / "v1.0" / "schema.json"
    return json.loads(path.read_text())


@pytest.fixture
def valid_eating_output() -> dict:
    """A valid eating event for testing."""
    return {
        "schema_version": "1.0",
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat",
            "breed_estimate": "british_shorthair",
            "id_tag": "grey_shorthair_medium",
            "id_confidence": 0.83,
            "action": {
                "primary": "eating",
                "distribution": {
                    "eating": 0.76, "drinking": 0.00,
                    "sniffing_only": 0.14, "leaving_bowl": 0.05,
                    "sitting_idle": 0.03, "other": 0.02,
                },
            },
            "eating_metrics": {
                "speed": {"fast": 0.08, "normal": 0.71, "slow": 0.21},
                "engagement": 0.74,
                "abandoned_midway": 0.12,
            },
            "mood": {"alertness": 0.28, "anxiety": 0.09, "engagement": 0.76},
            "body_signals": {"posture": "relaxed", "ear_position": "forward"},
            "anomaly_signals": {
                "vomit_gesture": 0.02, "food_rejection": 0.09,
                "excessive_sniffing": 0.16, "lethargy": 0.04, "aggression": 0.01,
            },
        },
        "bowl": {
            "food_fill_ratio": 0.42,
            "water_fill_ratio": None,
            "food_type_visible": "dry",
        },
        "scene": {
            "lighting": "bright",
            "image_quality": "clear",
            "confidence_overall": 0.85,
        },
        "narrative": "灰色英短以正常速度进食干粮，碗内余粮约42%，状态放松。",
    }


@pytest.fixture
def valid_no_pet_output() -> dict:
    """A valid no-pet event for testing."""
    return {
        "schema_version": "1.0",
        "pet_present": False,
        "pet_count": 0,
        "pet": None,
        "bowl": {
            "food_fill_ratio": 0.88,
            "water_fill_ratio": None,
            "food_type_visible": "dry",
        },
        "scene": {
            "lighting": "infrared_night",
            "image_quality": "clear",
            "confidence_overall": 0.92,
        },
        "narrative": "无宠物，碗内余粮充足约88%，夜视模式。",
    }


@pytest.fixture
def vision_event_dict() -> dict:
    """Minimal valid PetFeederEvent dict matching schema v1.0 (see src/pet_schema/models.py)."""
    return {
        "schema_version": "1.0",
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat",
            "breed_estimate": "unknown",
            "id_tag": "grey_shorthair_medium",
            "id_confidence": 0.85,
            "action": {
                "primary": "eating",
                "distribution": {
                    "eating": 1.0,
                    "drinking": 0.0,
                    "sniffing_only": 0.0,
                    "leaving_bowl": 0.0,
                    "sitting_idle": 0.0,
                    "other": 0.0,
                },
            },
            "eating_metrics": {
                "speed": {"fast": 0.0, "normal": 1.0, "slow": 0.0},
                "engagement": 0.8,
                "abandoned_midway": 0.0,
            },
            "mood": {"alertness": 0.7, "anxiety": 0.1, "engagement": 0.8},
            "body_signals": {"posture": "relaxed", "ear_position": "forward"},
            "anomaly_signals": {
                "vomit_gesture": 0.0,
                "food_rejection": 0.0,
                "excessive_sniffing": 0.0,
                "lethargy": 0.0,
                "aggression": 0.0,
            },
        },
        "bowl": {
            "food_fill_ratio": 0.5,
            "water_fill_ratio": None,
            "food_type_visible": "dry",
        },
        "scene": {
            "lighting": "bright",
            "image_quality": "clear",
            "confidence_overall": 0.9,
        },
        "narrative": "a cat eating from a metal bowl",
    }
