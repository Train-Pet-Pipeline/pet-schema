from datetime import datetime

import pytest
from pydantic import ValidationError

from pet_schema import (
    Annotation,
    ClassifierAnnotation,
    DpoPair,
    HumanAnnotation,
    LLMAnnotation,
    RuleAnnotation,
)

BASE_KW = dict(
    annotation_id="a1",
    target_id="f1",
    annotator_id="qwen2vl-7b",
    modality="vision",
    schema_version="2.1.0",
    created_at=datetime(2026, 4, 21),
    storage_uri=None,
)


# LLMAnnotation
def test_llm_roundtrip():
    src = LLMAnnotation(
        **BASE_KW,
        prompt_hash="abc",
        raw_response="...",
        parsed_output={"event": "eat"},
    )
    rt = LLMAnnotation.model_validate_json(src.model_dump_json())
    assert rt == src


def test_llm_missing_required():
    with pytest.raises(ValidationError):
        LLMAnnotation(**BASE_KW, prompt_hash="a", raw_response="b")  # 缺 parsed_output


def test_llm_modality_enum():
    with pytest.raises(ValidationError):
        LLMAnnotation(
            **{**BASE_KW, "modality": "infrared"},
            prompt_hash="a",
            raw_response="b",
            parsed_output={},
        )


# ClassifierAnnotation
def test_classifier_roundtrip():
    src = ClassifierAnnotation(
        **{**BASE_KW, "modality": "audio", "annotator_id": "audio-cnn-v1"},
        predicted_class="bark",
        class_probs={"bark": 0.9, "meow": 0.1},
        logits=[1.2, -0.3],
    )
    rt = ClassifierAnnotation.model_validate_json(src.model_dump_json())
    assert rt == src


def test_classifier_logits_optional():
    ClassifierAnnotation(
        **BASE_KW,
        predicted_class="x",
        class_probs={"x": 1.0},
        logits=None,
    )


def test_classifier_missing_required():
    with pytest.raises(ValidationError):
        ClassifierAnnotation(**BASE_KW, predicted_class="x")  # 缺 class_probs


# RuleAnnotation
def test_rule_roundtrip():
    src = RuleAnnotation(**BASE_KW, rule_id="threshold_v1", rule_output={"passed": True})
    rt = RuleAnnotation.model_validate_json(src.model_dump_json())
    assert rt == src


# HumanAnnotation
def test_human_roundtrip():
    src = HumanAnnotation(
        **{**BASE_KW, "annotator_id": "alice"},
        reviewer="alice",
        decision="accept",
        notes=None,
    )
    rt = HumanAnnotation.model_validate_json(src.model_dump_json())
    assert rt == src


# Discriminator routing
def test_annotation_discriminator_routes_llm():
    from pydantic import TypeAdapter

    ta = TypeAdapter(Annotation)
    obj = ta.validate_python(
        {
            **BASE_KW,
            "annotator_type": "llm",
            "prompt_hash": "h",
            "raw_response": "r",
            "parsed_output": {},
            "created_at": "2026-04-21T00:00:00",
        }
    )
    assert isinstance(obj, LLMAnnotation)


def test_annotation_discriminator_routes_classifier():
    from pydantic import TypeAdapter

    ta = TypeAdapter(Annotation)
    obj = ta.validate_python(
        {
            **BASE_KW,
            "annotator_type": "classifier",
            "predicted_class": "x",
            "class_probs": {"x": 1.0},
            "logits": None,
            "created_at": "2026-04-21T00:00:00",
        }
    )
    assert isinstance(obj, ClassifierAnnotation)


def test_annotation_discriminator_unknown_type_rejected():
    from pydantic import TypeAdapter

    ta = TypeAdapter(Annotation)
    with pytest.raises(ValidationError):
        ta.validate_python({**BASE_KW, "annotator_type": "vlm"})  # 旧名应被拒


# DpoPair
def test_dpo_pair_roundtrip():
    src = DpoPair(
        pair_id="p1",
        chosen_annotation_id="a1",
        rejected_annotation_id="a2",
        target_id="f1",
        modality="vision",
        preference_source="human",
        reason="reviewer preferred clearer label",
        created_at=datetime(2026, 4, 21),
        schema_version="2.1.0",
    )
    rt = DpoPair.model_validate_json(src.model_dump_json())
    assert rt == src


def test_dpo_pair_reason_optional():
    src = DpoPair(
        pair_id="p2",
        chosen_annotation_id="a1",
        rejected_annotation_id="a2",
        target_id="f1",
        modality="vision",
        preference_source="rule",
        reason=None,
        created_at=datetime(2026, 4, 21),
        schema_version="2.1.0",
    )
    assert src.reason is None


def test_dpo_pair_missing_preference_source():
    with pytest.raises(ValidationError):
        DpoPair(
            pair_id="p1",
            chosen_annotation_id="a1",
            rejected_annotation_id="a2",
            target_id="f1",
            modality="vision",
            created_at=datetime(2026, 4, 21),
            schema_version="2.1.0",
        )
