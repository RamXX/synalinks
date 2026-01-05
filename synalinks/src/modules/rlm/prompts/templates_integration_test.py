"""Integration tests for model-specific prompt templates with real LLMs."""

import pytest

from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator
from synalinks.src.modules.rlm.prompts.templates import PROMPT_TEMPLATES


class TestPromptTemplateAutoDetection:
    """Integration tests for automatic prompt template selection."""

    def test_zai_model_gets_zai_template(self):
        """Test that zai model automatically gets zai template."""
        lm = LanguageModel(model="zai/glm-4.7")
        gen = RecursiveGenerator(
            language_model=lm,
        )

        # Verify auto-detection selected zai template
        assert gen.prompt_template == PROMPT_TEMPLATES["zai"]
        assert "# Task Instructions" in gen.prompt_template
        assert "# Available Tools" in gen.prompt_template

    def test_groq_model_gets_groq_template(self):
        """Test that groq model automatically gets groq template."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        gen = RecursiveGenerator(
            language_model=lm,
        )

        # Verify auto-detection selected groq template
        assert gen.prompt_template == PROMPT_TEMPLATES["groq"]
        assert "# REPL Functions" in gen.prompt_template

    def test_groq_llama_model_gets_groq_template(self):
        """Test that groq/llama model gets groq template."""
        lm = LanguageModel(model="groq/llama-3.1-70b")
        gen = RecursiveGenerator(
            language_model=lm,
        )

        # Verify auto-detection selected groq template
        assert gen.prompt_template == PROMPT_TEMPLATES["groq"]

    def test_unknown_model_gets_default_template(self):
        """Test that unknown model gets default template."""
        lm = LanguageModel(model="openai/gpt-4")
        gen = RecursiveGenerator(
            language_model=lm,
        )

        # Verify auto-detection fell back to default
        assert gen.prompt_template == PROMPT_TEMPLATES["default"]
        assert "# REPL Environment" in gen.prompt_template

    def test_explicit_template_overrides_auto_detection(self):
        """Test that explicit template parameter overrides auto-detection."""
        lm = LanguageModel(model="zai/glm-4.7")
        custom_template = "Custom template: {{ instructions }}"

        gen = RecursiveGenerator(
            language_model=lm,
            prompt_template=custom_template,
        )

        # Verify explicit template was used instead of auto-detected
        assert gen.prompt_template == custom_template
        assert gen.prompt_template != PROMPT_TEMPLATES["zai"]


class TestPromptTemplateSerialization:
    """Test that prompt templates serialize correctly."""

    def test_prompt_template_in_config(self):
        """Test that prompt_template is included in get_config()."""
        lm = LanguageModel(model="zai/glm-4.7")
        gen = RecursiveGenerator(
            language_model=lm,
        )

        config = gen.get_config()

        # Verify prompt_template is in config
        assert "prompt_template" in config
        assert config["prompt_template"] == PROMPT_TEMPLATES["zai"]

    def test_custom_template_serialization(self):
        """Test that custom template serializes correctly."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        custom_template = "Custom: {{ instructions }}"

        gen = RecursiveGenerator(
            language_model=lm,
            prompt_template=custom_template,
        )

        config = gen.get_config()

        # Verify custom template is preserved
        assert config["prompt_template"] == custom_template

    def test_prompt_template_roundtrip_serialization(self):
        """Test that prompt_template survives roundtrip serialization."""
        lm = LanguageModel(model="zai/glm-4.7")
        gen1 = RecursiveGenerator(
            language_model=lm,
        )

        # Serialize and deserialize
        config = gen1.get_config()
        gen2 = RecursiveGenerator.from_config(config)

        # Verify template is preserved
        assert gen2.prompt_template == gen1.prompt_template
        assert gen2.prompt_template == PROMPT_TEMPLATES["zai"]


class TestPromptTemplateIntegrationWithSeeds:
    """Test that templates integrate with seed_instructions for training."""

    def test_template_structure_supports_training(self):
        """Test that templates have structure suitable for optimization."""
        # All templates should have instructions variable for seed optimization
        for template_name, template in PROMPT_TEMPLATES.items():
            # Templates must have instructions placeholder
            assert (
                "{{ instructions }}" in template
            ), f"{template_name} missing instructions variable"

            # Templates should support examples for few-shot optimization
            assert (
                "{% if examples %}" in template
            ), f"{template_name} missing examples support"
            assert (
                "{% for example in examples %}" in template
            ), f"{template_name} missing example loop"

    def test_zai_template_optimized_for_structured_prompts(self):
        """Test that zai template has structure for clear prompts."""
        zai_template = PROMPT_TEMPLATES["zai"]

        # zai should have clear section markers
        assert "# Task Instructions" in zai_template
        assert "# Available Tools" in zai_template
        assert "## Approach" in zai_template

        # Should guide step-by-step reasoning
        assert "1." in zai_template or "step" in zai_template.lower()

    def test_groq_template_optimized_for_conciseness(self):
        """Test that groq template is concise and direct."""
        groq_template = PROMPT_TEMPLATES["groq"]

        # groq should be more concise than zai
        assert len(groq_template) < len(PROMPT_TEMPLATES["zai"])

        # Should still have essential components
        assert "llm_query" in groq_template
        assert "FINAL" in groq_template

    def test_default_template_general_purpose(self):
        """Test that default template works across model families."""
        default_template = PROMPT_TEMPLATES["default"]

        # Should have standard structure
        assert "# Instructions" in default_template
        assert "# REPL Environment" in default_template
        assert "llm_query(prompt: str) -> str" in default_template
        assert "FINAL(value)" in default_template


class TestMultiModelTemplateComparison:
    """Compare templates across different model families."""

    def test_different_models_get_different_templates(self):
        """Test that different model families get distinct templates."""
        lm_zai = LanguageModel(model="zai/glm-4.7")
        lm_groq = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen_zai = RecursiveGenerator(language_model=lm_zai)
        gen_groq = RecursiveGenerator(language_model=lm_groq)

        # Verify they got different templates
        assert gen_zai.prompt_template != gen_groq.prompt_template

        # Verify each got their expected template
        assert gen_zai.prompt_template == PROMPT_TEMPLATES["zai"]
        assert gen_groq.prompt_template == PROMPT_TEMPLATES["groq"]

    def test_sub_model_template_independent(self):
        """Test that sub_language_model doesn't affect prompt template selection."""
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveGenerator(
            language_model=lm_root,
            sub_language_model=lm_sub,
        )

        # Template should be selected based on root model, not sub
        assert gen.prompt_template == PROMPT_TEMPLATES["zai"]
        assert gen.prompt_template != PROMPT_TEMPLATES["groq"]

    def test_template_selection_with_different_groq_models(self):
        """Test that all groq models get groq template."""
        groq_models = [
            "groq/openai/gpt-oss-20b",
            "groq/llama-3.1-70b",
            "groq/mixtral-8x7b",
        ]

        for model in groq_models:
            lm = LanguageModel(model=model)
            gen = RecursiveGenerator(language_model=lm)

            # All groq models should get groq template
            assert (
                gen.prompt_template == PROMPT_TEMPLATES["groq"]
            ), f"Model {model} did not get groq template"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
