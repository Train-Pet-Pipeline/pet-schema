"""Tests for pet_schema.training_samples — SFTSample, DPOSample, ShareGPTTurn."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pet_schema.training_samples import (
    DPOSample,
    SFTSample,
    ShareGPTSFTSample,
    ShareGPTTurn,
)

# ---------------------------------------------------------------------------
# ShareGPTTurn
# ---------------------------------------------------------------------------


class TestShareGPTTurn:
    """Validate turn-level ShareGPT wire format."""

    def test_human_turn_valid(self) -> None:
        t = ShareGPTTurn(**{"from": "human", "value": "What is my cat doing?"})
        assert t.from_ == "human"
        assert t.value == "What is my cat doing?"

    def test_gpt_turn_valid(self) -> None:
        t = ShareGPTTurn(**{"from": "gpt", "value": "Your cat is eating."})
        assert t.from_ == "gpt"

    def test_system_turn_valid(self) -> None:
        t = ShareGPTTurn(**{"from": "system", "value": "You are a pet monitor."})
        assert t.from_ == "system"

    def test_observation_turn_valid(self) -> None:
        t = ShareGPTTurn(**{"from": "observation", "value": "tool result"})
        assert t.from_ == "observation"

    def test_function_call_turn_valid(self) -> None:
        t = ShareGPTTurn(**{"from": "function_call", "value": '{"name":"detect"}'})
        assert t.from_ == "function_call"

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShareGPTTurn(**{"from": "assistant", "value": "wrong role name"})

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShareGPTTurn(**{"from": "human", "value": "hi", "extra_field": "bad"})

    def test_model_validate_dict_with_alias(self) -> None:
        data = {"from": "human", "value": "hello"}
        t = ShareGPTTurn.model_validate(data)
        assert t.from_ == "human"

    def test_serialise_uses_alias(self) -> None:
        t = ShareGPTTurn(**{"from": "gpt", "value": "ok"})
        dumped = t.model_dump(by_alias=True)
        assert "from" in dumped
        assert dumped["from"] == "gpt"
        assert "from_" not in dumped


# ---------------------------------------------------------------------------
# ShareGPTSFTSample
# ---------------------------------------------------------------------------


class TestShareGPTSFTSample:
    """Validate ShareGPT conversations format for LLaMA-Factory consumption."""

    def _minimal(self) -> dict:
        return {
            "conversations": [
                {"from": "human", "value": "describe the scene"},
                {"from": "gpt", "value": "a cat is eating"},
            ]
        }

    def test_minimal_valid(self) -> None:
        s = ShareGPTSFTSample.model_validate(self._minimal())
        assert len(s.conversations) == 2
        assert s.conversations[0].from_ == "human"

    def test_empty_conversations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShareGPTSFTSample.model_validate({"conversations": []})

    def test_extra_fields_rejected(self) -> None:
        data = self._minimal()
        data["unknown_field"] = "bad"
        with pytest.raises(ValidationError):
            ShareGPTSFTSample.model_validate(data)

    def test_optional_fields_default_none(self) -> None:
        s = ShareGPTSFTSample.model_validate(self._minimal())
        assert s.system is None
        assert s.tools is None
        assert s.sample_id is None
        assert s.source_target_id is None
        assert s.annotator_id is None

    def test_with_lineage_metadata(self) -> None:
        data = self._minimal()
        data["sample_id"] = "target-001"
        data["annotator_id"] = "llm-v1"
        s = ShareGPTSFTSample.model_validate(data)
        assert s.sample_id == "target-001"
        assert s.annotator_id == "llm-v1"

    def test_model_validate_json(self) -> None:
        """Simulate parsing a real JSONL line in LLaMA-Factory conversations format."""
        line = json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "What is the cat doing?"},
                    {"from": "gpt", "value": "The cat is drinking water."},
                ],
                "sample_id": "abc-123",
            }
        )
        s = ShareGPTSFTSample.model_validate_json(line)
        assert s.conversations[1].value == "The cat is drinking water."


# ---------------------------------------------------------------------------
# SFTSample (pet-annotation flat export format)
# ---------------------------------------------------------------------------


class TestSFTSample:
    """Validate flat SFT format produced by pet_annotation.export.sft_dpo.to_sft_samples."""

    def _minimal(self) -> dict:
        return {
            "sample_id": "target-001",
            "annotator_id": "llm-gpt4o",
            "annotator_type": "llm",
            "input": "s3://bucket/frame001.jpg",
            "output": '{"scene": {"confidence_overall": 0.95}}',
        }

    def test_minimal_valid(self) -> None:
        s = SFTSample.model_validate(self._minimal())
        assert s.sample_id == "target-001"
        assert s.annotator_type == "llm"
        assert s.storage_uri is None

    def test_with_storage_uri(self) -> None:
        data = self._minimal()
        data["storage_uri"] = "s3://bucket/frame001.jpg"
        s = SFTSample.model_validate(data)
        assert s.storage_uri == "s3://bucket/frame001.jpg"

    def test_invalid_annotator_type_rejected(self) -> None:
        data = self._minimal()
        data["annotator_type"] = "unknown"
        with pytest.raises(ValidationError):
            SFTSample.model_validate(data)

    def test_extra_fields_rejected(self) -> None:
        data = self._minimal()
        data["extra"] = "bad"
        with pytest.raises(ValidationError):
            SFTSample.model_validate(data)

    def test_all_annotator_types_accepted(self) -> None:
        for atype in ("llm", "classifier", "rule", "human"):
            data = self._minimal()
            data["annotator_type"] = atype
            s = SFTSample.model_validate(data)
            assert s.annotator_type == atype

    def test_model_validate_json_real_line(self) -> None:
        """Simulate parsing a real pet-annotation JSONL export line."""
        line = json.dumps(
            {
                "sample_id": "target-002",
                "annotator_id": "llm-claude3",
                "annotator_type": "llm",
                "input": "s3://bucket/frame002.jpg",
                "output": '{"scene":{"label":"cat_drinking","confidence_overall":0.9}}',
                "storage_uri": "s3://bucket/frame002.jpg",
            }
        )
        s = SFTSample.model_validate_json(line)
        assert s.sample_id == "target-002"
        assert s.storage_uri == "s3://bucket/frame002.jpg"


# ---------------------------------------------------------------------------
# DPOSample (pet-annotation flat DPO export format)
# ---------------------------------------------------------------------------


class TestDPOSample:
    """Validate flat DPO format produced by pet_annotation.export.sft_dpo.to_dpo_pairs."""

    def _minimal(self) -> dict:
        return {
            "sample_id": "target-003",
            "chosen": '{"scene":{"label":"cat_eating","confidence_overall":0.95}}',
            "rejected": '{"scene":{"label":"cat_eating","confidence_overall":0.7}}',
            "chosen_annotator_id": "llm-gpt4o",
            "rejected_annotator_id": "llm-gpt3",
        }

    def test_minimal_valid(self) -> None:
        d = DPOSample.model_validate(self._minimal())
        assert d.sample_id == "target-003"
        assert d.storage_uri is None

    def test_with_storage_uri(self) -> None:
        data = self._minimal()
        data["storage_uri"] = "s3://bucket/frame003.jpg"
        d = DPOSample.model_validate(data)
        assert d.storage_uri == "s3://bucket/frame003.jpg"

    def test_extra_fields_rejected(self) -> None:
        data = self._minimal()
        data["unexpected"] = "bad"
        with pytest.raises(ValidationError):
            DPOSample.model_validate(data)

    def test_chosen_and_rejected_required(self) -> None:
        data = self._minimal()
        del data["chosen"]
        with pytest.raises(ValidationError):
            DPOSample.model_validate(data)

    def test_model_validate_json_real_line(self) -> None:
        """Simulate parsing a real pet-annotation DPO JSONL export line."""
        line = json.dumps(
            {
                "sample_id": "target-004",
                "chosen": "The cat is eating dry food.",
                "rejected": "A cat.",
                "chosen_annotator_id": "llm-v2",
                "rejected_annotator_id": "llm-v1",
                "storage_uri": "s3://bucket/frame004.jpg",
            }
        )
        d = DPOSample.model_validate_json(line)
        assert d.sample_id == "target-004"
        assert d.chosen_annotator_id == "llm-v2"
