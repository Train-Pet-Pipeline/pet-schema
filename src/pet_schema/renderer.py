"""Prompt renderer — loads and renders system/user prompts from versioned templates."""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

VERSIONS_DIR = Path(__file__).parent / "versions"


def render_prompt(
    version: str = "1.0", few_shot: bool = True
) -> tuple[str, str]:
    """Render system and user prompts for a given schema version.

    Args:
        version: Schema version directory to load from.
        few_shot: Whether to inject few-shot examples into the user prompt.

    Returns:
        Tuple of (system_prompt, user_prompt).

    Raises:
        FileNotFoundError: If the version directory or required files don't exist.
    """
    version_dir = VERSIONS_DIR / f"v{version}"
    system_path = version_dir / "prompt_system.txt"
    user_template_path = version_dir / "prompt_user.jinja2"
    examples_path = version_dir / "few_shot_examples.json"

    if not version_dir.exists():
        msg = f"Schema 版本目录不存在: {version_dir}"
        raise FileNotFoundError(msg)
    if not system_path.exists():
        msg = f"System prompt 不存在: {system_path}"
        raise FileNotFoundError(msg)
    if not user_template_path.exists():
        msg = f"User prompt 模板不存在: {user_template_path}"
        raise FileNotFoundError(msg)

    system_prompt = system_path.read_text(encoding="utf-8")

    template = Template(user_template_path.read_text(encoding="utf-8"))

    few_shot_examples = ""
    if few_shot and examples_path.exists():
        examples = json.loads(examples_path.read_text(encoding="utf-8"))
        few_shot_examples = json.dumps(examples, ensure_ascii=False, indent=2)

    user_prompt = template.render(few_shot_examples=few_shot_examples if few_shot else "")

    return system_prompt, user_prompt
