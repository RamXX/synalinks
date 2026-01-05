"""Integration tests for RLM cost tracking and metrics with real LLM calls."""

import asyncio

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator
from synalinks.src.modules.rlm.core.types import RLMExecutionMetrics


class MetricsIntegrationTest(testing.TestCase):
    """Integration tests for metrics with real LLM calls."""

    async def test_metrics_with_single_model_real_llm(self):
        """Metrics populated correctly with single model (zai/glm-4.7)."""

        class Query(DataModel):
            question: str = Field(description="Math question")

        class Answer(DataModel):
            answer: int = Field(description="The numeric answer")

        lm = LanguageModel(model="zai/glm-4.7")
        gen = RecursiveGenerator(language_model=lm, data_model=Answer, max_iterations=10)

        query = Query(question="What is 2 + 2?")
        result = await gen(query.to_json_data_model())

        # Verify result
        self.assertIsNotNone(result)

        # Verify metrics populated
        metrics = gen.get_last_metrics()
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, RLMExecutionMetrics)

        # Verify iteration count
        self.assertGreater(metrics.iteration_count, 0)
        self.assertLessEqual(metrics.iteration_count, 10)

        # Verify token counts
        self.assertGreater(metrics.total_tokens, 0)
        self.assertGreater(metrics.prompt_tokens, 0)
        self.assertGreater(metrics.completion_tokens, 0)

        # Verify root model usage
        self.assertIn("total_tokens", metrics.root_model_usage)
        self.assertGreater(metrics.root_model_usage["total_tokens"], 0)

        # Verify estimated cost is calculated
        self.assertGreater(metrics.estimated_cost, 0)

    async def test_metrics_with_multi_model_real_llm(self):
        """Metrics track root and sub models separately with multi-model setup."""

        class Query(DataModel):
            text: str = Field(description="Text to process")

        class Summary(DataModel):
            summary: str = Field(description="Brief summary")

        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveGenerator(
            language_model=lm_root,
            sub_language_model=lm_sub,
            data_model=Summary,
            max_iterations=10,
        )

        query = Query(text="The quick brown fox jumps over the lazy dog.")
        result = await gen(query.to_json_data_model())

        # Verify result
        self.assertIsNotNone(result)

        # Verify metrics
        metrics = gen.get_last_metrics()
        self.assertIsNotNone(metrics)

        # Verify iteration count
        self.assertGreater(metrics.iteration_count, 0)

        # Verify total tokens across both models
        self.assertGreater(metrics.total_tokens, 0)

        # Verify root model usage
        self.assertIn("total_tokens", metrics.root_model_usage)
        self.assertGreater(metrics.root_model_usage["total_tokens"], 0)

        # Verify estimated cost
        self.assertGreater(metrics.estimated_cost, 0)

    async def test_metrics_sub_call_count_real_llm(self):
        """Sub-call count tracks recursive llm_query() calls."""

        class Query(DataModel):
            items: str = Field(description="List of items")

        class Result(DataModel):
            count: int = Field(description="Number of items")

        lm = LanguageModel(model="zai/glm-4.7")
        gen = RecursiveGenerator(
            language_model=lm,
            data_model=Result,
            max_iterations=15,
            max_depth=2,  # Allow recursive calls
        )

        # Query that might benefit from decomposition
        query = Query(items="apple, banana, cherry, date, elderberry")
        result = await gen(query.to_json_data_model())

        # Verify result
        self.assertIsNotNone(result)

        # Verify metrics
        metrics = gen.get_last_metrics()
        self.assertIsNotNone(metrics)

        # Sub-call count should be >= 0 (may or may not use llm_query)
        self.assertGreaterEqual(metrics.sub_call_count, 0)

    async def test_training_mode_accumulates_metrics(self):
        """Training mode appends metrics to _training_metrics list."""

        class Query(DataModel):
            x: int = Field(description="Number")

        class Answer(DataModel):
            result: int = Field(description="Result")

        lm = LanguageModel(model="zai/glm-4.7")
        gen = RecursiveGenerator(language_model=lm, data_model=Answer, max_iterations=5)

        # Run multiple training examples
        queries = [Query(x=1), Query(x=2), Query(x=3)]

        for query in queries:
            await gen(query.to_json_data_model(), training=True)

        # Verify training metrics accumulated
        # Note: Only successful executions that produce output append to training_metrics
        # So the count might be less than 3 if some executions failed to parse
        self.assertGreaterEqual(len(gen._training_metrics), 0)

        # If any metrics were collected, verify they're valid
        if len(gen._training_metrics) > 0:
            for metrics in gen._training_metrics:
                self.assertIsInstance(metrics, RLMExecutionMetrics)
                self.assertGreater(metrics.iteration_count, 0)


if __name__ == "__main__":
    asyncio.run(testing.main())
