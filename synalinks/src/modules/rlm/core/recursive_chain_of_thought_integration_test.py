"""Integration tests for RecursiveChainOfThought with real LLM calls."""

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.recursive_chain_of_thought import (
    RecursiveChainOfThought,
)


class Query(DataModel):
    """Test input model."""

    query: str = Field(description="User query")


class Answer(DataModel):
    """Test output model."""

    answer: str = Field(description="The answer")


class RecursiveChainOfThoughtIntegrationTest(testing.TestCase):
    """E2E tests for RecursiveChainOfThought with real LLM calls."""

    async def test_basic_cot_with_thinking_fields(self):
        """RecursiveChainOfThought generates output with thinking fields."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveChainOfThought(
            data_model=Answer,
            language_model=lm,
            k=2,  # 2 thinking steps
            max_iterations=10,
        )

        query = Query(query="What is 2+2?").to_json_data_model()
        result = await gen(query)

        # Verify result has thinking fields and answer
        self.assertIsNotNone(result)
        self.assertIn("thinking", result)
        self.assertIn("thinking_1", result)  # Second thinking field
        self.assertIn("answer", result)

        # Thinking fields should be populated
        self.assertIsInstance(result["thinking"], str)
        self.assertGreater(len(result["thinking"]), 0)

    async def test_multi_model_cot_architecture(self):
        """RecursiveChainOfThought works with multi-model setup."""
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveChainOfThought(
            data_model=Answer,
            language_model=lm_root,
            sub_language_model=lm_sub,
            k=3,  # 3 thinking steps
            max_iterations=10,
            max_depth=1,
        )

        query = Query(query="Calculate 5+7").to_json_data_model()
        result = await gen(query)

        # Verify result has all thinking fields
        self.assertIsNotNone(result)
        self.assertIn("thinking", result)
        self.assertIn("thinking_1", result)
        self.assertIn("thinking_2", result)
        self.assertIn("answer", result)

    async def test_cot_with_trajectory_logging(self):
        """RecursiveChainOfThought logs trajectory when enabled."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveChainOfThought(
            data_model=Answer,
            language_model=lm,
            k=1,
            max_iterations=10,
            enable_trajectory_logging=True,
        )

        query = Query(query="What is 10-3?").to_json_data_model()
        result = await gen(query)

        # Verify trajectory was logged
        trajectory = gen.get_last_trajectory()
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.root_model, "zai/glm-4.7")
        self.assertGreater(len(trajectory.iterations), 0)

        # Verify result has thinking field
        self.assertIsNotNone(result)
        self.assertIn("thinking", result)
        self.assertIn("answer", result)

    async def test_cot_without_trajectory_logging(self):
        """RecursiveChainOfThought does not log when disabled."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveChainOfThought(
            data_model=Answer,
            language_model=lm,
            k=1,
            max_iterations=10,
            enable_trajectory_logging=False,  # Disabled
        )

        query = Query(query="What is 3*4?").to_json_data_model()
        await gen(query)

        # Trajectory should be None when disabled
        trajectory = gen.get_last_trajectory()
        self.assertIsNone(trajectory)

    async def test_cot_with_code_execution(self):
        """RecursiveChainOfThought can execute code and think step-by-step."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveChainOfThought(
            data_model=Answer,
            language_model=lm,
            k=2,
            max_iterations=15,
            enable_trajectory_logging=True,
        )

        # Query that should trigger code execution
        query = Query(query="Calculate the sum of 123 and 456").to_json_data_model()
        result = await gen(query)

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIn("thinking", result)
        self.assertIn("thinking_1", result)
        self.assertIn("answer", result)

        # Check trajectory for code execution
        trajectory = gen.get_last_trajectory()
        self.assertIsNotNone(trajectory)

        # Verify some code was executed (structure check)
        has_code = False
        for iteration in trajectory.iterations:
            if len(iteration.code_blocks) > 0:
                has_code = True
                break
        self.assertIsInstance(has_code, bool)

    async def test_cot_multi_model_with_recursive_calls(self):
        """RecursiveChainOfThought multi-model handles recursive calls."""
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveChainOfThought(
            data_model=Answer,
            language_model=lm_root,
            sub_language_model=lm_sub,
            k=2,
            max_iterations=10,
            max_depth=2,  # Allow sub-calls
            enable_trajectory_logging=True,
        )

        query = Query(query="What is 8 divided by 2?").to_json_data_model()
        result = await gen(query)

        # Verify result
        self.assertIsNotNone(result)
        self.assertIn("thinking", result)
        self.assertIn("thinking_1", result)
        self.assertIn("answer", result)

        # Verify trajectory tracks both models
        trajectory = gen.get_last_trajectory()
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.root_model, "zai/glm-4.7")
        self.assertEqual(trajectory.sub_model, "groq/openai/gpt-oss-20b")
        self.assertEqual(trajectory.max_depth, 2)
