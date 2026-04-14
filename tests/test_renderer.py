"""Tests for prompt renderer."""
import pytest

from pet_schema.renderer import render_prompt


class TestRenderPrompt:
    def test_returns_system_and_user(self):
        system, user = render_prompt()
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0

    def test_system_prompt_contains_key_instructions(self):
        system, _ = render_prompt()
        assert "JSON" in system
        assert "不拟人化" in system
        assert "80" in system

    def test_user_prompt_with_few_shot(self):
        _, user = render_prompt(few_shot=True)
        assert "参考示例" in user
        assert "british_shorthair" in user

    def test_user_prompt_without_few_shot(self):
        _, user = render_prompt(few_shot=False)
        assert "参考示例" not in user

    def test_invalid_version_raises(self):
        with pytest.raises(FileNotFoundError):
            render_prompt(version="99.0")
