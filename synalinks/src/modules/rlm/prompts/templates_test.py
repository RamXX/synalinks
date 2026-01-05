"""Tests for model-specific prompt templates."""

import pytest

from synalinks.src.modules.rlm.prompts.templates import PROMPT_TEMPLATES
from synalinks.src.modules.rlm.prompts.templates import get_prompt_template


class TestPromptTemplatesRegistry:
    """Test the PROMPT_TEMPLATES registry."""

    def test_registry_contains_default(self):
        """Test that registry contains default template."""
        assert "default" in PROMPT_TEMPLATES
        assert isinstance(PROMPT_TEMPLATES["default"], str)
        assert len(PROMPT_TEMPLATES["default"]) > 0

    def test_registry_contains_zai(self):
        """Test that registry contains zai template."""
        assert "zai" in PROMPT_TEMPLATES
        assert isinstance(PROMPT_TEMPLATES["zai"], str)
        assert len(PROMPT_TEMPLATES["zai"]) > 0

    def test_registry_contains_groq(self):
        """Test that registry contains groq template."""
        assert "groq" in PROMPT_TEMPLATES
        assert isinstance(PROMPT_TEMPLATES["groq"], str)
        assert len(PROMPT_TEMPLATES["groq"]) > 0

    def test_templates_are_different(self):
        """Test that each template has unique content."""
        # Ensure templates are meaningfully different, not just copies
        default = PROMPT_TEMPLATES["default"]
        zai = PROMPT_TEMPLATES["zai"]
        groq = PROMPT_TEMPLATES["groq"]

        # All should be different from each other
        assert default != zai
        assert default != groq
        assert zai != groq

    def test_templates_contain_jinja_variables(self):
        """Test that templates contain required Jinja2 variables."""
        required_vars = [
            "{{ instructions }}",
            "{{ inputs_schema }}",
            "{{ outputs_schema }}",
        ]

        for template_name, template in PROMPT_TEMPLATES.items():
            for var in required_vars:
                assert var in template, f"{template_name} missing {var}"

    def test_templates_contain_repl_functions(self):
        """Test that templates mention REPL functions."""
        # All templates should mention the core REPL functions
        for template_name, template in PROMPT_TEMPLATES.items():
            assert "llm_query" in template, f"{template_name} missing llm_query"
            assert "FINAL" in template, f"{template_name} missing FINAL"

    def test_zai_template_has_structured_format(self):
        """Test that zai template has structured formatting."""
        zai = PROMPT_TEMPLATES["zai"]

        # zai template should have clear section markers
        assert "# Task Instructions" in zai
        assert "# Available Tools" in zai
        assert "## Approach" in zai

    def test_groq_template_is_concise(self):
        """Test that groq template is more concise."""
        groq = PROMPT_TEMPLATES["groq"]

        # groq template should be relatively concise
        # Check for minimal formatting markers (fewer # headers)
        assert groq.count("#") < PROMPT_TEMPLATES["zai"].count("#")


class TestGetPromptTemplate:
    """Test the get_prompt_template function."""

    def test_none_returns_default(self):
        """Test that None model returns default template."""
        template = get_prompt_template(None)
        assert template == PROMPT_TEMPLATES["default"]

    def test_zai_model_returns_zai_template(self):
        """Test that zai model returns zai template."""
        template = get_prompt_template("zai/glm-4.7")
        assert template == PROMPT_TEMPLATES["zai"]

    def test_groq_model_returns_groq_template(self):
        """Test that groq model returns groq template."""
        template = get_prompt_template("groq/openai/gpt-oss-20b")
        assert template == PROMPT_TEMPLATES["groq"]

    def test_groq_llama_model_returns_groq_template(self):
        """Test that groq/llama model returns groq template."""
        template = get_prompt_template("groq/llama-3.1-70b")
        assert template == PROMPT_TEMPLATES["groq"]

    def test_unknown_model_returns_default(self):
        """Test that unknown model returns default template."""
        template = get_prompt_template("unknown/model-xyz")
        assert template == PROMPT_TEMPLATES["default"]

    def test_model_without_slash_returns_default(self):
        """Test that model without slash returns default if not in registry."""
        template = get_prompt_template("somemodel")
        assert template == PROMPT_TEMPLATES["default"]

    def test_empty_string_returns_default(self):
        """Test that empty string returns default template."""
        template = get_prompt_template("")
        assert template == PROMPT_TEMPLATES["default"]

    def test_prefix_extraction(self):
        """Test that prefix extraction works correctly."""
        # Test various model formats
        test_cases = [
            ("zai/glm-4.7", PROMPT_TEMPLATES["zai"]),
            ("zai/some-other-model", PROMPT_TEMPLATES["zai"]),
            ("groq/openai/gpt-oss-20b", PROMPT_TEMPLATES["groq"]),
            ("groq/llama-3.1-70b", PROMPT_TEMPLATES["groq"]),
            ("openai/gpt-4", PROMPT_TEMPLATES["default"]),
            ("anthropic/claude-3", PROMPT_TEMPLATES["default"]),
        ]

        for model, expected_template in test_cases:
            result = get_prompt_template(model)
            assert result == expected_template, f"Failed for model: {model}"

    def test_case_sensitivity(self):
        """Test that model prefix matching is case-sensitive."""
        # "ZAI" (uppercase) should not match "zai" template
        template = get_prompt_template("ZAI/glm-4.7")
        assert template == PROMPT_TEMPLATES["default"]

        # Correct case should work
        template = get_prompt_template("zai/glm-4.7")
        assert template == PROMPT_TEMPLATES["zai"]


class TestTemplateContent:
    """Test template content and structure."""

    def test_default_template_completeness(self):
        """Test that default template has all expected sections."""
        template = PROMPT_TEMPLATES["default"]

        # Check for all major sections
        assert "# Instructions" in template
        assert "# Input Schema" in template
        assert "# Output Schema" in template
        assert "# Examples" in template
        assert "# REPL Environment" in template

    def test_zai_template_completeness(self):
        """Test that zai template has all expected sections."""
        template = PROMPT_TEMPLATES["zai"]

        # Check for all major sections
        assert "# Task Instructions" in template
        assert "# Input Format" in template
        assert "# Expected Output Format" in template
        assert "# Reference Examples" in template
        assert "# Available Tools" in template
        assert "## Approach" in template

    def test_groq_template_completeness(self):
        """Test that groq template has all expected sections."""
        template = PROMPT_TEMPLATES["groq"]

        # Check for all major sections
        assert "# Instructions" in template
        assert "# Input Schema" in template
        assert "# Output Schema" in template
        assert "# Examples" in template
        assert "# REPL Functions" in template

    def test_templates_support_conditional_sections(self):
        """Test that templates use Jinja2 conditionals for optional sections."""
        # All templates should have conditional blocks
        for template_name, template in PROMPT_TEMPLATES.items():
            # Check for if blocks around optional sections
            assert "{% if inputs_schema %}" in template
            assert "{% endif %}" in template
            assert "{% if outputs_schema %}" in template
            assert "{% if examples %}" in template

    def test_templates_support_example_loops(self):
        """Test that templates loop over examples."""
        for template_name, template in PROMPT_TEMPLATES.items():
            assert "{% for example in examples %}" in template
            assert "{% endfor %}" in template
            # Should reference example inputs and outputs
            assert "example[0]" in template
            assert "example[1]" in template


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
