"""pet-schema → pet-ota manifest.json adapter."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pet_schema.model_card import ModelCard
from pet_schema.version import SCHEMA_VERSION


def build_manifest(cards: Sequence[ModelCard]) -> dict[str, Any]:
    """Build a manifest.json-ready dict from a list of ModelCards (pure)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "models": [c.to_manifest_entry() for c in cards],
    }
