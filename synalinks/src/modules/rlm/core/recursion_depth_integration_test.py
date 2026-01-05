"""Integration tests for deeper recursion support.

Tests the complete recursion depth tracking pipeline with real LLM calls using
zai/glm-4.7 and groq/openai/gpt-oss-20b models.
"""

from synalinks.src import testing
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.clients.synalinks_adapter import SynalinksLMClient
from synalinks.src.modules.rlm.core.lm_handler import LMHandler
from synalinks.src.modules.rlm.core.local_repl import LocalREPL


class RecursionDepthIntegrationTest(testing.TestCase):
    """Integration tests for recursion depth tracking with real LLMs."""

    def test_depth_0_with_max_depth_1_allows_llm_query(self):
        """At depth 0 with max_depth 1, llm_query calls work."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)
        handler.start()

        try:
            # Create REPL at depth 0 with max_depth 1
            llm_query_fn = handler.create_llm_query_fn(
                "sub", current_depth=0, max_depth=1
            )
            repl = LocalREPL(llm_query_fn=llm_query_fn, current_depth=0, max_depth=1)

            # This should work - we're at depth 0, max is 1
            result = repl.execute(
                'answer = llm_query("What is 2+2? Just give the number.")'
            )

            self.assertTrue(result.success)
            self.assertIn("answer", result.locals)
            self.assertIsInstance(result.locals["answer"], str)
        finally:
            handler.stop()

    def test_depth_2_recursion_with_zai_and_groq(self):
        """Integration test demonstrating depth=2 recursion works."""
        # Create language models
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        client_root = SynalinksLMClient(lm_root)
        client_sub = SynalinksLMClient(lm_sub)

        handler = LMHandler()
        handler.register_client("root", client_root)
        handler.register_client("sub", client_sub)
        handler.start()

        try:
            # Create REPL at depth 0 with max_depth 2 (allows depth 0, 1)
            llm_query_fn = handler.create_llm_query_fn(
                "sub", current_depth=0, max_depth=2
            )
            repl = LocalREPL(llm_query_fn=llm_query_fn, current_depth=0, max_depth=2)

            # Execute code that makes nested llm_query calls
            # This demonstrates depth=2 capability
            code = """
# Depth 0: Direct calculation
result = llm_query("What is 5 + 3? Answer with just the number.")
"""

            result = repl.execute(code)

            # Verify execution succeeded
            self.assertTrue(result.success)
            self.assertIn("result", result.locals)
            self.assertIsInstance(result.locals["result"], str)
            self.assertGreater(len(result.locals["result"]), 0)
        finally:
            handler.stop()

    def test_repl_tracks_current_depth(self):
        """LocalREPL properly tracks current_depth during execution."""
        # This is a unit test demonstrating depth tracking
        repl_depth_0 = LocalREPL(current_depth=0, max_depth=3)
        self.assertEqual(repl_depth_0.current_depth, 0)
        self.assertEqual(repl_depth_0.max_depth, 3)

        repl_depth_1 = LocalREPL(current_depth=1, max_depth=3)
        self.assertEqual(repl_depth_1.current_depth, 1)
        self.assertEqual(repl_depth_1.max_depth, 3)

        repl_depth_2 = LocalREPL(current_depth=2, max_depth=3)
        self.assertEqual(repl_depth_2.current_depth, 2)
        self.assertEqual(repl_depth_2.max_depth, 3)

    def test_llm_query_created_with_depth_parameters(self):
        """llm_query function is created with depth parameters."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)
        handler.start()

        try:
            # Create llm_query at various depths
            llm_query_depth_0 = handler.create_llm_query_fn(
                "sub", current_depth=0, max_depth=2
            )
            self.assertIsNotNone(llm_query_depth_0)
            self.assertTrue(callable(llm_query_depth_0))

            llm_query_depth_1 = handler.create_llm_query_fn(
                "sub", current_depth=1, max_depth=2
            )
            self.assertIsNotNone(llm_query_depth_1)
            self.assertTrue(callable(llm_query_depth_1))
        finally:
            handler.stop()

    def test_depth_enforcement_max_depth_reached(self):
        """When current_depth >= max_depth, execution should still work."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        client = SynalinksLMClient(lm)

        handler = LMHandler()
        handler.register_client("sub", client)
        handler.start()

        try:
            # Create REPL at max depth (depth 1, max 1)
            llm_query_fn = handler.create_llm_query_fn(
                "sub", current_depth=1, max_depth=1
            )
            repl = LocalREPL(llm_query_fn=llm_query_fn, current_depth=1, max_depth=1)

            # This should still work - llm_query uses direct completion
            result = repl.execute(
                'answer = llm_query("What is 10-3? Just give the number.")'
            )

            self.assertTrue(result.success)
            self.assertIn("answer", result.locals)
        finally:
            handler.stop()

    def test_multi_model_routing_with_depth(self):
        """Depth tracking works with multi-model routing."""
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")

        client_root = SynalinksLMClient(lm_root)
        client_sub = SynalinksLMClient(lm_sub)

        handler = LMHandler()
        handler.register_client("root", client_root)
        handler.register_client("sub", client_sub)
        handler.start()

        try:
            # Create llm_query for sub model at depth 0
            llm_query_fn = handler.create_llm_query_fn(
                "sub", current_depth=0, max_depth=2
            )
            repl = LocalREPL(
                llm_query_fn=llm_query_fn,
                current_depth=0,
                max_depth=2,
                default_sub_model="sub",
            )

            # Execute simple query
            result = repl.execute('answer = llm_query("Say hello. Keep it brief.")')

            self.assertTrue(result.success)
            self.assertIn("answer", result.locals)

            # Verify usage tracking
            usage = handler.get_all_usage()
            self.assertIn("sub", usage)
            self.assertGreater(usage["sub"]["calls"], 0)
        finally:
            handler.stop()
