"""Training data sample contracts.

SFTSample + DPOSample define the wire format between pet-annotation's exporter
and pet-train's trainer plugins. Both producer (pet-annotation) and consumer
(pet-train) validate against these models to prevent silent JSONL field drift.

**Two formats are represented here:**

1. ``SFTSample`` — flat export format produced by ``pet_annotation.export.sft_dpo.to_sft_samples``.
   Fields: ``sample_id``, ``annotator_id``, ``annotator_type``, ``input``, ``output``,
   ``storage_uri``.

2. ``ShareGPTTurn`` / ``ShareGPTSFTSample`` — ShareGPT conversations format consumed by
   LLaMA-Factory's ``SharegptDatasetConverter``.  Field names match the LLaMA-Factory
   ``DatasetAttr`` defaults (``role_tag="from"``, ``content_tag="value"``).

.. note::
   pet-annotation currently emits ``SFTSample`` (flat format).  LLaMA-Factory's ShareGPT
   dataset loader expects ``ShareGPTSFTSample`` (conversations list format).  These two
   formats are **not** directly interchangeable — a conversion step is required between
   them.  This mismatch is a known gap (flagged Phase 5, 2026-04-23); pet-annotation v2.1.1
   export validator and pet-train consumer-side validator each use the relevant model.

Added v3.2.0 for cross-repo JSONL contract (per user 2026-04-23 Phase 5 decision).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ShareGPTTurn(BaseModel):
    """One conversation turn in ShareGPT-style format.

    Field names match LLaMA-Factory's ``DatasetAttr`` defaults:
    ``role_tag="from"`` and ``content_tag="value"``.
    Valid role values mirror LLaMA-Factory tag defaults:
    ``user_tag="human"``, ``assistant_tag="gpt"``, ``system_tag="system"``,
    ``observation_tag="observation"``, ``function_tag="function_call"``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: Literal["human", "gpt", "system", "observation", "function_call"] = Field(
        alias="from"
    )
    value: str


class ShareGPTSFTSample(BaseModel):
    """SFT sample in ShareGPT conversations format for LLaMA-Factory consumption.

    This is the format LLaMA-Factory's ``SharegptDatasetConverter`` expects when
    ``formatting="sharegpt"`` is used.  The top-level key ``conversations`` maps to
    ``DatasetAttr.messages`` (default ``"conversations"``).
    """

    model_config = ConfigDict(extra="forbid")

    conversations: list[ShareGPTTurn] = Field(min_length=1)
    system: str | None = None
    tools: str | None = None
    # Optional lineage (ignored by LLaMA-Factory, useful for debugging):
    sample_id: str | None = None
    source_target_id: str | None = None
    annotator_id: str | None = None


class SFTSample(BaseModel):
    """Supervised fine-tuning sample in flat format as produced by pet-annotation.

    This is the wire format emitted by ``pet_annotation.export.sft_dpo.to_sft_samples``.
    Fields exactly match the dict keys that function yields.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    annotator_id: str
    annotator_type: Literal["llm", "classifier", "rule", "human"]
    input: str
    output: str
    storage_uri: str | None = None
