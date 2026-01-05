"""Tests for RecursiveGenerator."""

from unittest.mock import MagicMock
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator


class RecursiveGeneratorTest(testing.TestCase):
    """Tests for RecursiveGenerator module."""

    def test_instantiates_without_error(self):
        """RecursiveGenerator instantiates successfully."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        gen = RecursiveGenerator(language_model=mock_lm)
        self.assertIsNotNone(gen)

    def test_requires_language_model(self):
        """RecursiveGenerator requires language_model parameter."""
        with self.assertRaises(ValueError):
            RecursiveGenerator()

    def test_supports_separate_sub_language_model(self):
        """RecursiveGenerator accepts separate sub_language_model."""
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"
        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        gen = RecursiveGenerator(
            language_model=mock_lm_root, sub_language_model=mock_lm_sub
        )

        self.assertEqual(gen.language_model, mock_lm_root)
        self.assertEqual(gen.sub_language_model, mock_lm_sub)

    def test_sub_language_model_defaults_to_language_model(self):
        """sub_language_model defaults to language_model when not provided."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        gen = RecursiveGenerator(language_model=mock_lm)

        # sub_language_model should default to language_model
        self.assertEqual(gen.language_model, mock_lm)
        self.assertEqual(gen.sub_language_model, mock_lm)
        self.assertIs(gen.sub_language_model, gen.language_model)

    def test_serialization(self):
        """RecursiveGenerator supports get_config/from_config."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "test"

        gen = RecursiveGenerator(
            language_model=mock_lm, max_iterations=5, name="test_gen"
        )

        config = gen.get_config()
        self.assertEqual(config["max_iterations"], 5)
        self.assertEqual(config["name"], "test_gen")

    def test_serialization_with_both_language_models(self):
        """get_config() serializes both language_model and sub_language_model."""
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"
        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        gen = RecursiveGenerator(
            language_model=mock_lm_root,
            sub_language_model=mock_lm_sub,
            max_iterations=10,
            name="multi_model_gen",
        )

        config = gen.get_config()

        # Both models should be in config
        self.assertIn("language_model", config)
        self.assertIn("sub_language_model", config)
        self.assertEqual(config["language_model"], mock_lm_root)
        self.assertEqual(config["sub_language_model"], mock_lm_sub)
        self.assertEqual(config["max_iterations"], 10)
        self.assertEqual(config["name"], "multi_model_gen")

    def test_from_config_restores_both_language_models(self):
        """from_config() restores both language models correctly."""
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"
        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        gen1 = RecursiveGenerator(
            language_model=mock_lm_root,
            sub_language_model=mock_lm_sub,
            max_iterations=15,
        )

        config = gen1.get_config()
        gen2 = RecursiveGenerator.from_config(config)

        # Verify restored state
        self.assertEqual(gen2.language_model, mock_lm_root)
        self.assertEqual(gen2.sub_language_model, mock_lm_sub)
        self.assertEqual(gen2.max_iterations, 15)

    @patch("litellm.acompletion")
    async def test_minimal_e2e_flow(self, mock_litellm):
        """Integration test: minimal e2e flow works."""
        # Mock LLM response
        mock_litellm.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }

        # Create minimal input
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveGenerator(language_model=lm)

        # Create simple input using DataModel
        from synalinks.src.backend import DataModel
        from synalinks.src.backend import Field

        class TestQuery(DataModel):
            query: str = Field(description="Test query")

        input_data = TestQuery(query="test")

        # Call should complete without error
        result = await gen(input_data)
        self.assertIsNotNone(result)
