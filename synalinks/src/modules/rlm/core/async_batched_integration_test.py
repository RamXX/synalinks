"""Integration and performance tests for async batched execution.

Tests the complete async batching pipeline with real LLM calls using
zai/glm-4.7 and groq/openai/gpt-oss-20b models.
"""

import time

from synalinks.src import testing
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.clients.synalinks_adapter import SynalinksLMClient
from synalinks.src.modules.rlm.core.lm_handler import LMHandler
from synalinks.src.modules.rlm.core.local_repl import LocalREPL


class AsyncBatchedIntegrationTest(testing.TestCase):
    """Integration tests for async batched execution with real LLMs."""

    async def test_acompletion_batched_with_real_llm(self):
        """acompletion_batched works with real LLM (groq/openai/gpt-oss-20b)."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)

        prompts = [
            "What is 2+2? Answer with just the number.",
            "What is 3+3? Answer with just the number.",
            "What is 5+5? Answer with just the number.",
        ]

        results = await handler.acompletion_batched(prompts, "sub")

        # All prompts should succeed
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    async def test_acompletion_batched_performance_5x_speedup(self):
        """Batched execution is ~5x faster than sequential for 5 calls."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)

        prompts = [
            f"Count to {i}. Just give the count."
            for i in range(1, 6)  # 5 prompts
        ]

        # Time sequential execution
        start_seq = time.time()
        sequential_results = []
        for prompt in prompts:
            result = await client.acompletion(prompt)
            sequential_results.append(result)
        sequential_time = time.time() - start_seq

        # Time batched execution
        start_batch = time.time()
        batched_results = await handler.acompletion_batched(prompts, "sub")
        batched_time = time.time() - start_batch

        # Verify correctness
        self.assertEqual(len(batched_results), 5)
        for result in batched_results:
            self.assertIsInstance(result, str)

        # Verify speedup (should be ~5x faster, allow 3x for variance)
        speedup = sequential_time / batched_time
        print(
            f"\nPerformance: Sequential={sequential_time:.2f}s, "
            f"Batched={batched_time:.2f}s, Speedup={speedup:.2f}x"
        )
        self.assertGreaterEqual(
            speedup,
            3.0,
            f"Expected ~5x speedup, got {speedup:.2f}x "
            f"(seq={sequential_time:.2f}s, batch={batched_time:.2f}s)",
        )

    async def test_partial_errors_dont_fail_batch(self):
        """Individual errors don't fail entire batch."""
        from unittest.mock import AsyncMock
        from unittest.mock import MagicMock

        # Create a mock client that fails on second call
        call_count = [0]

        async def mock_acompletion(prompt):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError(f"Simulated error on prompt {call_count[0]}")
            return f"Response to: {prompt}"

        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(side_effect=mock_acompletion)

        handler = LMHandler()
        handler.register_client("test", mock_client)

        prompts = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        results = await handler.acompletion_batched(prompts, "test")

        # Verify partial success
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0], "Response to: Q1")
        self.assertIsInstance(results[1], ValueError)
        self.assertEqual(results[2], "Response to: Q3")
        self.assertEqual(results[3], "Response to: Q4")
        self.assertEqual(results[4], "Response to: Q5")

    def test_llm_query_batched_via_tcp_socket(self):
        """llm_query_batched works via TCP socket with real LLM."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)
        handler.start()

        try:
            llm_query_batched = handler.create_llm_query_batched_fn("sub")

            prompts = [
                "Say 'A'",
                "Say 'B'",
                "Say 'C'",
            ]

            results = llm_query_batched(prompts)

            self.assertEqual(len(results), 3)
            for result in results:
                # Should be strings (not exceptions)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
        finally:
            handler.stop()

    def test_repl_with_llm_query_batched(self):
        """LocalREPL can execute llm_query_batched with real LLM."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)
        handler.start()

        try:
            llm_query_batched = handler.create_llm_query_batched_fn("sub")
            repl = LocalREPL(llm_query_batched_fn=llm_query_batched)

            code = """
prompts = ['What is 1+1?', 'What is 2+2?']
results = llm_query_batched(prompts)
print(f'Got {len(results)} results')
"""

            result = repl.execute(code)

            self.assertTrue(result.success)
            self.assertIn("Got 2 results", result.stdout)
            self.assertIn("results", result.locals)
            self.assertEqual(len(result.locals["results"]), 2)
        finally:
            handler.stop()

    async def test_multi_model_batched_execution(self):
        """Can batch with different models (root vs sub)."""
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        client_root = SynalinksLMClient(lm_root)
        client_sub = SynalinksLMClient(lm_sub)

        handler = LMHandler()
        handler.register_client("root", client_root)
        handler.register_client("sub", client_sub)

        prompts = ["Say 'hello'", "Say 'world'"]

        # Batch with root model
        results_root = await handler.acompletion_batched(prompts, "root")
        self.assertEqual(len(results_root), 2)

        # Batch with sub model
        results_sub = await handler.acompletion_batched(prompts, "sub")
        self.assertEqual(len(results_sub), 2)

        # Both should succeed
        for result in results_root:
            self.assertIsInstance(result, str)
        for result in results_sub:
            self.assertIsInstance(result, str)

    async def test_empty_batch_returns_empty_list(self):
        """Batched call with empty prompts list returns empty list."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)

        results = await handler.acompletion_batched([], "sub")
        self.assertEqual(results, [])

    async def test_single_prompt_batch_works(self):
        """Batched call with single prompt works correctly."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)

        results = await handler.acompletion_batched(["Say 'test'"], "sub")

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], str)
        self.assertGreater(len(results[0]), 0)
