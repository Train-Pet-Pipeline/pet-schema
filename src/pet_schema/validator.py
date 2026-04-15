"""VLM output validator — JSON Schema + extra business constraints."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

VERSIONS_DIR = Path(__file__).parent / "versions"


@dataclass
class ValidationResult:
    """Result of validating a VLM output string."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_output(json_str: str, version: str = "1.0") -> ValidationResult:
    """Validate a VLM JSON output string.

    Runs JSON Schema validation followed by extra business-rule checks.

    Args:
        json_str: Raw JSON string from VLM output.
        version: Schema version to validate against.

    Returns:
        ValidationResult with valid=True if all checks pass.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return ValidationResult(valid=False, errors=[f"JSON 解析失败: {e}"])

    schema_path = VERSIONS_DIR / f"v{version}" / "schema.json"
    if not schema_path.exists():
        return ValidationResult(
            valid=False, errors=[f"Schema 版本 {version} 不存在: {schema_path}"]
        )
    schema = json.loads(schema_path.read_text())

    errors: list[str] = []
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema 验证失败: {e.message}")

    errors.extend(_extra_validations(data))

    warnings: list[str] = []
    scene = data.get("scene", {})
    if isinstance(scene.get("confidence_overall"), float) and scene["confidence_overall"] < 0.5:
        warnings.append(f"confidence_overall 偏低: {scene['confidence_overall']}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _extra_validations(data: dict[str, Any]) -> list[str]:
    """Business-rule validations that JSON Schema cannot express.

    Checks:
        - pet_present / pet field consistency
        - action.distribution sum = 1.0 +/- 0.01
        - eating_metrics.speed sum = 1.0 +/- 0.01
        - action.primary matches highest probability in distribution
        - narrative length <= 80 chars

    Args:
        data: Parsed JSON dict.

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []

    if data.get("pet_present") and data.get("pet") is None:
        errors.append("pet_present=true 但 pet 字段为 null")
    if not data.get("pet_present") and data.get("pet") is not None:
        errors.append("pet_present=false 但 pet 字段非 null")

    pet = data.get("pet")
    if pet:
        dist = pet.get("action", {}).get("distribution", {})
        if dist:
            total = sum(float(v) for v in dist.values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"action.distribution 求和 {total:.4f} 超出 1.0±0.01"
                )

        primary = pet.get("action", {}).get("primary")
        if dist and primary:
            max_val = max(float(v) for v in dist.values())
            max_keys = [k for k, v in dist.items() if float(v) == max_val]
            if primary not in max_keys:
                top = ", ".join(max_keys)
                errors.append(
                    f"action.primary '{primary}' 不是 distribution 中最高概率的 key: {top}"
                )

        speed = pet.get("eating_metrics", {}).get("speed", {})
        if speed:
            total = sum(float(v) for v in speed.values())
            # 当宠物不在进食时（action.primary 非 eating/drinking），speed 分布全 0 合法
            is_eating = primary in ("eating", "drinking")
            if total > 0 or is_eating:
                if abs(total - 1.0) > 0.01:
                    errors.append(
                        f"eating_metrics.speed 求和 {total:.4f} 超出 1.0±0.01"
                    )

    narrative = data.get("narrative", "")
    if len(narrative) > 80:
        errors.append(f"narrative 长度 {len(narrative)} 字超过 80 字限制")

    return errors
