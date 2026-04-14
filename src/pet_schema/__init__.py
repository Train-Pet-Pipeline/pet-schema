"""pet-schema: VLM output schema definition, validation, and prompt rendering.

Public API:
    validate_output -- Validate a VLM JSON output string against the schema.
    render_prompt -- Render system and user prompts for a given schema version.
"""
from pet_schema.renderer import render_prompt
from pet_schema.validator import validate_output

__all__ = ["validate_output", "render_prompt"]
