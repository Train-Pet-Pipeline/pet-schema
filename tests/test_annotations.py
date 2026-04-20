from datetime import datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from pet_schema.annotations import (
    Annotation,
    AudioAnnotation,
    DpoPair,
    VisionAnnotation,
)
from pet_schema.models import PetFeederEvent


def test_vision_annotation_wraps_pet_feeder_event(vision_annotation_dict):
    a = VisionAnnotation.model_validate(vision_annotation_dict)
    assert a.modality == "vision"
    assert isinstance(a.parsed, PetFeederEvent)


def test_audio_annotation_requires_probs_dict():
    with pytest.raises(ValidationError):
        AudioAnnotation(
            annotation_id="a1",
            sample_id="s1",
            annotator_type="cnn",
            annotator_id="audio_cnn_v1",
            created_at=datetime(2026, 4, 20),
            schema_version="2.0.0",
            predicted_class="bark",
            class_probs="not-a-dict",  # type: ignore[arg-type]
        )


def test_annotation_union_discriminator(vision_annotation_dict, audio_annotation_dict):
    adapter = TypeAdapter(Annotation)
    assert isinstance(adapter.validate_python(vision_annotation_dict), VisionAnnotation)
    assert isinstance(adapter.validate_python(audio_annotation_dict), AudioAnnotation)


def test_dpo_pair_validates():
    p = DpoPair(
        pair_id="p1",
        chosen_annotation_id="a1",
        rejected_annotation_id="a2",
        preference_source="human",
        reason="chosen is cleaner",
    )
    assert p.preference_source == "human"


def test_dpo_pair_rejects_bad_source():
    with pytest.raises(ValidationError):
        DpoPair(
            pair_id="p1",
            chosen_annotation_id="a1",
            rejected_annotation_id="a2",
            preference_source="random",  # not in literal
        )


def test_vision_annotation_rejects_wrong_modality(vision_annotation_dict):
    bad = dict(vision_annotation_dict)
    bad["modality"] = "audio"
    with pytest.raises(ValidationError):
        VisionAnnotation.model_validate(bad)
