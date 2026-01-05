"""Tests for trajectory logging functionality."""

import json

from synalinks.src import testing
from synalinks.src.modules.rlm.core.types import RLMExecutionMetrics
from synalinks.src.modules.rlm.core.types import RLMIteration
from synalinks.src.modules.rlm.core.types import RLMSubCall
from synalinks.src.modules.rlm.core.types import RLMTrajectory


class RLMSubCallTest(testing.TestCase):
    """Tests for RLMSubCall dataclass."""

    def test_creates_sub_call(self):
        """RLMSubCall creates instance with all fields."""
        sub_call = RLMSubCall(
            model="groq/openai/gpt-oss-20b",
            prompt="What is 2+2?",
            response="4",
            depth=1,
        )

        self.assertEqual(sub_call.model, "groq/openai/gpt-oss-20b")
        self.assertEqual(sub_call.prompt, "What is 2+2?")
        self.assertEqual(sub_call.response, "4")
        self.assertEqual(sub_call.depth, 1)
        self.assertIsNone(sub_call.error)

    def test_sub_call_with_error(self):
        """RLMSubCall can store error information."""
        sub_call = RLMSubCall(
            model="groq/openai/gpt-oss-20b",
            prompt="Invalid query",
            response="",
            depth=1,
            error="Timeout error",
        )

        self.assertEqual(sub_call.error, "Timeout error")

    def test_to_dict_conversion(self):
        """RLMSubCall.to_dict() produces correct dictionary."""
        sub_call = RLMSubCall(
            model="groq/openai/gpt-oss-20b",
            prompt="Test prompt",
            response="Test response",
            depth=2,
        )

        result = sub_call.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["model"], "groq/openai/gpt-oss-20b")
        self.assertEqual(result["prompt"], "Test prompt")
        self.assertEqual(result["response"], "Test response")
        self.assertEqual(result["depth"], 2)
        self.assertIsNone(result["error"])


class RLMIterationTest(testing.TestCase):
    """Tests for RLMIteration dataclass."""

    def test_creates_iteration(self):
        """RLMIteration creates instance with required fields."""
        iteration = RLMIteration(
            iteration=0, prompt="System prompt", response="Model response"
        )

        self.assertEqual(iteration.iteration, 0)
        self.assertEqual(iteration.prompt, "System prompt")
        self.assertEqual(iteration.response, "Model response")
        self.assertEqual(iteration.code_blocks, [])
        self.assertEqual(iteration.execution_results, [])
        self.assertEqual(iteration.sub_calls, [])
        self.assertIsNone(iteration.final_answer)

    def test_iteration_with_code_blocks(self):
        """RLMIteration stores code blocks."""
        iteration = RLMIteration(
            iteration=1,
            prompt="Execute code",
            response="Running...",
            code_blocks=["x = 1 + 2", "print(x)"],
        )

        self.assertEqual(len(iteration.code_blocks), 2)
        self.assertEqual(iteration.code_blocks[0], "x = 1 + 2")

    def test_iteration_with_execution_results(self):
        """RLMIteration stores execution results."""
        exec_result = {
            "stdout": "3\n",
            "stderr": "",
            "exception": None,
            "final_answer": None,
        }
        iteration = RLMIteration(
            iteration=1,
            prompt="Execute",
            response="Done",
            execution_results=[exec_result],
        )

        self.assertEqual(len(iteration.execution_results), 1)
        self.assertEqual(iteration.execution_results[0]["stdout"], "3\n")

    def test_iteration_with_sub_calls(self):
        """RLMIteration stores sub-calls."""
        sub_call = RLMSubCall(
            model="groq/openai/gpt-oss-20b",
            prompt="Sub-query",
            response="Sub-response",
            depth=1,
        )
        iteration = RLMIteration(
            iteration=1,
            prompt="Main prompt",
            response="Main response",
            sub_calls=[sub_call],
        )

        self.assertEqual(len(iteration.sub_calls), 1)
        self.assertEqual(iteration.sub_calls[0].model, "groq/openai/gpt-oss-20b")

    def test_to_dict_conversion(self):
        """RLMIteration.to_dict() produces correct dictionary."""
        sub_call = RLMSubCall(
            model="groq/openai/gpt-oss-20b",
            prompt="Sub",
            response="Response",
            depth=1,
        )
        iteration = RLMIteration(
            iteration=0,
            prompt="Test prompt",
            response="Test response",
            code_blocks=["code"],
            sub_calls=[sub_call],
            final_answer={"answer": "42"},
        )

        result = iteration.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["iteration"], 0)
        self.assertEqual(result["prompt"], "Test prompt")
        self.assertEqual(result["response"], "Test response")
        self.assertEqual(result["code_blocks"], ["code"])
        self.assertEqual(len(result["sub_calls"]), 1)
        self.assertEqual(result["final_answer"], {"answer": "42"})


class RLMTrajectoryTest(testing.TestCase):
    """Tests for RLMTrajectory dataclass."""

    def test_creates_empty_trajectory(self):
        """RLMTrajectory creates with default values."""
        trajectory = RLMTrajectory()

        self.assertEqual(trajectory.iterations, [])
        self.assertEqual(trajectory.total_iterations, 0)
        self.assertEqual(trajectory.root_model, "")
        self.assertEqual(trajectory.sub_model, "")
        self.assertEqual(trajectory.max_iterations, 30)
        self.assertEqual(trajectory.max_depth, 1)
        self.assertFalse(trajectory.success)
        self.assertIsNone(trajectory.error)

    def test_creates_trajectory_with_config(self):
        """RLMTrajectory creates with specified configuration."""
        trajectory = RLMTrajectory(
            root_model="zai/glm-4.7",
            sub_model="groq/openai/gpt-oss-20b",
            max_iterations=10,
            max_depth=3,
        )

        self.assertEqual(trajectory.root_model, "zai/glm-4.7")
        self.assertEqual(trajectory.sub_model, "groq/openai/gpt-oss-20b")
        self.assertEqual(trajectory.max_iterations, 10)
        self.assertEqual(trajectory.max_depth, 3)

    def test_trajectory_with_iterations(self):
        """RLMTrajectory stores multiple iterations."""
        iter1 = RLMIteration(iteration=0, prompt="Prompt 1", response="Response 1")
        iter2 = RLMIteration(iteration=1, prompt="Prompt 2", response="Response 2")
        trajectory = RLMTrajectory(
            iterations=[iter1, iter2], total_iterations=2, success=True
        )

        self.assertEqual(len(trajectory.iterations), 2)
        self.assertEqual(trajectory.total_iterations, 2)
        self.assertTrue(trajectory.success)

    def test_to_dict_conversion(self):
        """RLMTrajectory.to_dict() produces correct dictionary."""
        iteration = RLMIteration(iteration=0, prompt="Test", response="Response")
        trajectory = RLMTrajectory(
            iterations=[iteration],
            total_iterations=1,
            root_model="zai/glm-4.7",
            sub_model="groq/openai/gpt-oss-20b",
            max_iterations=30,
            max_depth=1,
            success=True,
        )

        result = trajectory.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["iterations"]), 1)
        self.assertEqual(result["total_iterations"], 1)
        self.assertEqual(result["root_model"], "zai/glm-4.7")
        self.assertEqual(result["sub_model"], "groq/openai/gpt-oss-20b")
        self.assertTrue(result["success"])

    def test_to_json_produces_valid_json(self):
        """RLMTrajectory.to_json() produces valid JSON string."""
        iteration = RLMIteration(iteration=0, prompt="Test", response="Response")
        trajectory = RLMTrajectory(
            iterations=[iteration],
            total_iterations=1,
            root_model="zai/glm-4.7",
            success=True,
        )

        json_str = trajectory.to_json()

        # Should be parseable as JSON
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["total_iterations"], 1)
        self.assertEqual(parsed["root_model"], "zai/glm-4.7")
        self.assertTrue(parsed["success"])

    def test_to_markdown_produces_readable_output(self):
        """RLMTrajectory.to_markdown() produces markdown string."""
        iteration = RLMIteration(
            iteration=0,
            prompt="Test prompt",
            response="Test response",
            code_blocks=["x = 42"],
        )
        trajectory = RLMTrajectory(
            iterations=[iteration],
            total_iterations=1,
            root_model="zai/glm-4.7",
            sub_model="groq/openai/gpt-oss-20b",
            success=True,
        )

        markdown = trajectory.to_markdown()

        self.assertIn("# RLM Execution Trajectory", markdown)
        self.assertIn("**Root Model**: zai/glm-4.7", markdown)
        self.assertIn("**Sub Model**: groq/openai/gpt-oss-20b", markdown)
        self.assertIn("**Total Iterations**: 1/30", markdown)
        self.assertIn("**Success**: True", markdown)
        self.assertIn("## Iteration 0", markdown)
        self.assertIn("### Prompt", markdown)
        self.assertIn("### Response", markdown)
        self.assertIn("### Code Blocks", markdown)


class RLMExecutionMetricsTest(testing.TestCase):
    """Tests for RLMExecutionMetrics dataclass."""

    def test_creates_empty_metrics(self):
        """RLMExecutionMetrics creates with default values."""
        metrics = RLMExecutionMetrics()

        self.assertEqual(metrics.iteration_count, 0)
        self.assertEqual(metrics.sub_call_count, 0)
        self.assertEqual(metrics.total_tokens, 0)
        self.assertEqual(metrics.prompt_tokens, 0)
        self.assertEqual(metrics.completion_tokens, 0)

    def test_creates_metrics_with_values(self):
        """RLMExecutionMetrics creates with specified values."""
        metrics = RLMExecutionMetrics(
            iteration_count=5,
            sub_call_count=10,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
        )

        self.assertEqual(metrics.iteration_count, 5)
        self.assertEqual(metrics.sub_call_count, 10)
        self.assertEqual(metrics.total_tokens, 1000)
        self.assertEqual(metrics.prompt_tokens, 600)
        self.assertEqual(metrics.completion_tokens, 400)

    def test_estimated_cost_calculation(self):
        """RLMExecutionMetrics.estimated_cost calculates cost."""
        metrics = RLMExecutionMetrics(total_tokens=10000)

        cost = metrics.estimated_cost

        # 10000 tokens / 1000 * $0.0001 = $0.001
        self.assertAlmostEqual(cost, 0.001, places=4)
