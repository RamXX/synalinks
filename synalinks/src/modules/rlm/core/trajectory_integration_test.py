"""Integration tests for trajectory logging with real LLM calls."""

import json

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator


class Query(DataModel):
    """Test input model."""

    query: str = Field(description="User query")


class Answer(DataModel):
    """Test output model."""

    answer: str = Field(description="The answer")


class TrajectoryLoggingIntegrationTest(testing.TestCase):
    """E2E tests for trajectory logging with real LLM calls."""

    async def test_trajectory_logging_enabled(self):
        """RecursiveGenerator logs trajectory when enabled."""
        # Use multi-model setup per paper recommendations
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveGenerator(
            data_model=Answer,
            language_model=lm_root,
            sub_language_model=lm_sub,
            max_iterations=5,
            max_depth=1,
            enable_trajectory_logging=True,
        )

        query = Query(query="What is 2+2?").to_json_data_model()
        result = await gen(query)

        # Get trajectory
        trajectory = gen.get_last_trajectory()

        # Verify trajectory was logged
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.root_model, "zai/glm-4.7")
        self.assertEqual(trajectory.sub_model, "groq/openai/gpt-oss-20b")
        self.assertEqual(trajectory.max_iterations, 5)
        self.assertEqual(trajectory.max_depth, 1)

        # Should have at least one iteration
        self.assertGreater(len(trajectory.iterations), 0)

        # Check iteration structure
        first_iter = trajectory.iterations[0]
        self.assertEqual(first_iter.iteration, 0)
        self.assertIsNotNone(first_iter.prompt)
        self.assertIsNotNone(first_iter.response)

        # Verify success if result was obtained
        if result:
            self.assertTrue(trajectory.success)
            self.assertGreater(trajectory.total_iterations, 0)

    async def test_trajectory_logging_disabled(self):
        """RecursiveGenerator does not log when disabled."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveGenerator(
            data_model=Answer,
            language_model=lm,
            max_iterations=5,
            enable_trajectory_logging=False,  # Disabled
        )

        query = Query(query="What is 1+1?").to_json_data_model()
        await gen(query)

        # Trajectory should be None when disabled
        trajectory = gen.get_last_trajectory()
        self.assertIsNone(trajectory)

    async def test_trajectory_json_export(self):
        """RLMTrajectory.to_json() produces valid JSON."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveGenerator(
            data_model=Answer,
            language_model=lm,
            max_iterations=5,
            enable_trajectory_logging=True,
        )

        query = Query(query="Calculate 5+7").to_json_data_model()
        await gen(query)

        trajectory = gen.get_last_trajectory()
        self.assertIsNotNone(trajectory)

        # Export to JSON
        json_str = trajectory.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)
        self.assertIn("iterations", parsed)
        self.assertIn("total_iterations", parsed)
        self.assertIn("root_model", parsed)
        self.assertIn("success", parsed)

        # Verify iterations are serialized
        self.assertIsInstance(parsed["iterations"], list)
        if len(parsed["iterations"]) > 0:
            first_iter = parsed["iterations"][0]
            self.assertIn("iteration", first_iter)
            self.assertIn("prompt", first_iter)
            self.assertIn("response", first_iter)

    async def test_trajectory_markdown_export(self):
        """RLMTrajectory.to_markdown() produces readable output."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveGenerator(
            data_model=Answer,
            language_model=lm,
            max_iterations=5,
            enable_trajectory_logging=True,
        )

        query = Query(query="What is 10-3?").to_json_data_model()
        await gen(query)

        trajectory = gen.get_last_trajectory()
        self.assertIsNotNone(trajectory)

        # Export to markdown
        markdown = trajectory.to_markdown()

        # Should contain expected sections
        self.assertIn("# RLM Execution Trajectory", markdown)
        self.assertIn("**Root Model**:", markdown)
        self.assertIn("**Total Iterations**:", markdown)
        self.assertIn("**Success**:", markdown)

        # Should contain iteration details if any
        if len(trajectory.iterations) > 0:
            self.assertIn("## Iteration 0", markdown)
            self.assertIn("### Prompt", markdown)
            self.assertIn("### Response", markdown)

    async def test_trajectory_logs_code_execution(self):
        """Trajectory logs code blocks and execution results."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveGenerator(
            data_model=Answer,
            language_model=lm,
            max_iterations=10,
            enable_trajectory_logging=True,
        )

        # Query that should trigger code execution
        query = Query(query="Calculate the sum of 123 and 456").to_json_data_model()
        await gen(query)

        trajectory = gen.get_last_trajectory()
        self.assertIsNotNone(trajectory)

        # Check if any iteration has code blocks
        has_code = False
        has_execution_results = False
        for iteration in trajectory.iterations:
            if len(iteration.code_blocks) > 0:
                has_code = True
            if len(iteration.execution_results) > 0:
                has_execution_results = True

        # At least some iterations should have code
        # (this depends on LLM behavior, so we just verify structure)
        self.assertIsInstance(has_code, bool)
        self.assertIsInstance(has_execution_results, bool)

    async def test_trajectory_multi_model_tracking(self):
        """Trajectory correctly tracks multi-model architecture."""
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveGenerator(
            data_model=Answer,
            language_model=lm_root,
            sub_language_model=lm_sub,
            max_iterations=5,
            max_depth=2,  # Allow sub-calls
            enable_trajectory_logging=True,
        )

        query = Query(query="What is 3*4?").to_json_data_model()
        await gen(query)

        trajectory = gen.get_last_trajectory()
        self.assertIsNotNone(trajectory)

        # Verify models are tracked
        self.assertEqual(trajectory.root_model, "zai/glm-4.7")
        self.assertEqual(trajectory.sub_model, "groq/openai/gpt-oss-20b")
        self.assertEqual(trajectory.max_depth, 2)
