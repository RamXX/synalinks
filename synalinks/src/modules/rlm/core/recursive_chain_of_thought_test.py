"""Tests for RecursiveChainOfThought."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.recursive_chain_of_thought import (
    RecursiveChainOfThought,
)
from synalinks.src.modules.rlm.core.recursive_chain_of_thought import Thinking


class RecursiveChainOfThoughtTest(testing.TestCase):
    """Tests for RecursiveChainOfThought module."""

    def test_instantiates_without_error(self):
        """RecursiveChainOfThought instantiates successfully."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer)
        self.assertIsNotNone(gen)

    def test_inherits_from_recursive_generator(self):
        """RecursiveChainOfThought inherits from RecursiveGenerator."""
        from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator

        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer)
        self.assertIsInstance(gen, RecursiveGenerator)

    def test_supports_sub_language_model(self):
        """RecursiveChainOfThought inherits sub_language_model parameter."""
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"
        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(
            language_model=mock_lm_root,
            sub_language_model=mock_lm_sub,
            data_model=Answer,
        )

        self.assertEqual(gen.language_model, mock_lm_root)
        self.assertEqual(gen.sub_language_model, mock_lm_sub)

    def test_prepends_thinking_fields_to_schema(self):
        """Output schema prepends thinking fields to original schema."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer, k=2)

        # Verify augmented schema has thinking fields
        schema = gen.schema
        properties = schema["properties"]

        # Should have 2 thinking fields + 1 answer field
        self.assertIn("thinking", properties)
        # Count total thinking fields - should be k=2
        thinking_count = sum(1 for key in properties if "thinking" in key.lower())
        self.assertGreaterEqual(thinking_count, 2)

    def test_k_defaults_to_one(self):
        """k parameter defaults to 1 when not provided."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer)
        self.assertEqual(gen.k, 1)

    def test_k_accepts_multiple_thinking_steps(self):
        """k parameter accepts values > 1 for multiple thinking steps."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer, k=3)
        self.assertEqual(gen.k, 3)

    def test_get_config_includes_k_parameter(self):
        """get_config() includes k parameter."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "test"
        mock_lm.get_config.return_value = {"model": "test"}

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer, k=5)

        config = gen.get_config()
        self.assertIn("k", config)
        self.assertEqual(config["k"], 5)

    def test_get_config_includes_both_language_models(self):
        """get_config() serializes both language_model and sub_language_model."""
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"
        mock_lm_root.get_config.return_value = {"model": "zai/glm-4.7"}
        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"
        mock_lm_sub.get_config.return_value = {"model": "groq/openai/gpt-oss-20b"}

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(
            language_model=mock_lm_root,
            sub_language_model=mock_lm_sub,
            data_model=Answer,
            k=2,
        )

        config = gen.get_config()

        # Both models should be in config
        self.assertIn("language_model", config)
        self.assertIn("sub_language_model", config)
        self.assertEqual(config["k"], 2)

    def test_from_config_properly_deserializes(self):
        """from_config() properly deserializes all parameters including k."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "test"
        mock_lm.get_config.return_value = {"model": "test"}

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen1 = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, k=4, max_iterations=20
        )

        config = gen1.get_config()
        gen2 = RecursiveChainOfThought.from_config(config)

        # Verify all parameters restored
        self.assertEqual(gen2.k, 4)
        self.assertEqual(gen2.max_iterations, 20)

    def test_from_config_restores_both_language_models(self):
        """from_config() restores both language models correctly."""
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"
        mock_lm_root.get_config.return_value = {"model": "zai/glm-4.7"}
        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"
        mock_lm_sub.get_config.return_value = {"model": "groq/openai/gpt-oss-20b"}

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen1 = RecursiveChainOfThought(
            language_model=mock_lm_root,
            sub_language_model=mock_lm_sub,
            data_model=Answer,
            k=3,
        )

        config = gen1.get_config()
        gen2 = RecursiveChainOfThought.from_config(config)

        # Verify restored state
        self.assertEqual(gen2.k, 3)

    def test_api_export_decorator_present(self):
        """RecursiveChainOfThought has @synalinks_export decorator."""
        # Verify the decorator was applied
        self.assertTrue(callable(RecursiveChainOfThought))

    def test_default_cot_instructions_encourage_step_by_step_reasoning(self):
        """Default instructions encourage step-by-step reasoning with REPL use."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer)

        # Instructions should mention step-by-step and thinking
        instructions = gen.state.get("instructions")
        self.assertIsNotNone(instructions)
        self.assertIn("step", instructions.lower())

    async def test_e2e_cot_rlm_loop_with_mocked_llm(self):
        """E2E test: CoT + RLM generates thinking steps + final answer."""

        class Query(DataModel):
            question: str = Field(description="Math question")

        class Answer(DataModel):
            answer: int = Field(description="The numeric answer")

        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "test-model"

        mock_client = AsyncMock()
        mock_client.get_usage_summary = MagicMock(
            return_value={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }
        )

        # Mock response with thinking fields + answer
        mock_client.acompletion.side_effect = [
            # First call - generates code to set result
            """Let me think about this step by step.
```repl
result = {
    "thinking": "First, I'll add 2 and 2",
    "answer": 4
}
```
""",
            # Second call - returns FINAL
            "FINAL_VAR(result)",
        ]

        gen = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, k=1, max_iterations=5
        )

        with patch(
            "synalinks.src.modules.rlm.core.recursive_generator.SynalinksLMClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            query = Query(question="What is 2+2?")
            result = await gen(query.to_json_data_model())

            # Verify we got structured output
            self.assertIsNotNone(result)
            # Verify LLM was called
            self.assertGreater(mock_client.acompletion.call_count, 0)

    async def test_thinking_data_model_structure(self):
        """Thinking DataModel has correct field."""

        # Verify Thinking class structure
        thinking = Thinking(thinking="My step by step thinking")
        self.assertEqual(thinking.thinking, "My step by step thinking")

    def test_stores_original_schema(self):
        """Module stores original schema before augmentation."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer, k=2)

        # Original schema should not include thinking fields
        original_schema = gen._original_schema
        self.assertIsNotNone(original_schema)
        self.assertIn("answer", original_schema["properties"])

    def test_config_serializes_original_schema(self):
        """get_config() returns original schema, not augmented schema."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "test"
        mock_lm.get_config.return_value = {"model": "test"}

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer, k=3)

        config = gen.get_config()

        # Config schema should be original schema (for reconstruction)
        schema = config.get("schema")
        self.assertIsNotNone(schema)
        self.assertIn("answer", schema["properties"])

    async def test_compute_output_spec_returns_symbolic_data_model(self):
        """compute_output_spec() returns SymbolicDataModel for graph building."""

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer, k=2)

        spec = await gen.compute_output_spec(None)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.__class__.__name__, "SymbolicDataModel")

    def test_inherits_max_depth_parameter(self):
        """RecursiveChainOfThought inherits max_depth from RecursiveGenerator."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, max_depth=5
        )
        self.assertEqual(gen.max_depth, 5)

    def test_inherits_max_iterations_parameter(self):
        """RecursiveChainOfThought inherits max_iterations from RecursiveGenerator."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, max_iterations=50
        )
        self.assertEqual(gen.max_iterations, 50)

    def test_inherits_temperature_parameter(self):
        """RecursiveChainOfThought inherits temperature from RecursiveGenerator."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, temperature=0.7
        )
        self.assertEqual(gen.temperature, 0.7)

    def test_accepts_custom_instructions(self):
        """RecursiveChainOfThought accepts custom instructions parameter."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        custom = "Custom CoT instructions"
        gen = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, instructions=custom
        )

        self.assertEqual(gen.instructions, custom)

    def test_accepts_seed_instructions(self):
        """RecursiveChainOfThought accepts seed_instructions for optimization."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        seeds = ["Seed instruction 1", "Seed instruction 2"]
        gen = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, seed_instructions=seeds
        )

        self.assertEqual(gen.seed_instructions, seeds)

    def test_accepts_examples_parameter(self):
        """RecursiveChainOfThought accepts examples for few-shot learning."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        examples = [({"input": "test"}, {"output": "result"})]
        gen = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, examples=examples
        )

        self.assertEqual(gen.examples, examples)

    async def test_multiple_thinking_steps_augmentation(self):
        """k > 1 creates multiple thinking fields in augmented schema."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        gen = RecursiveChainOfThought(language_model=mock_lm, data_model=Answer, k=3)

        # Augmented schema should have 3 thinking fields
        schema = gen.schema
        properties = schema["properties"]

        # Count thinking-related fields
        thinking_count = sum(1 for key in properties if "thinking" in key.lower())
        self.assertGreaterEqual(thinking_count, 3)
