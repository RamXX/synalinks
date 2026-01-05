"""E2E tests for multi-model architecture routing."""

from unittest.mock import MagicMock

from synalinks.src import testing
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.clients.synalinks_adapter import SynalinksLMClient
from synalinks.src.modules.rlm.core.lm_handler import LMHandler
from synalinks.src.modules.rlm.core.local_repl import LocalREPL
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator


class MultiModelE2ETest(testing.TestCase):
    """E2E tests for multi-model routing and cost optimization."""

    def test_multi_model_routing_via_lm_handler(self):
        """E2E: LMHandler routes requests to correct client by name."""
        # Create mock LMs with different models
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"

        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        # Setup completion mocks
        async def mock_root_completion(messages):
            response = MagicMock()
            response.content = "Root model response"
            response.usage = {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
            }
            return response

        async def mock_sub_completion(messages):
            response = MagicMock()
            response.content = "Sub model response"
            response.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
            return response

        mock_lm_root.side_effect = mock_root_completion
        mock_lm_sub.side_effect = mock_sub_completion

        # Create clients
        client_root = SynalinksLMClient(mock_lm_root)
        client_sub = SynalinksLMClient(mock_lm_sub)

        # Create handler and register clients
        handler = LMHandler()
        handler.register_client("root", client_root)
        handler.register_client("sub", client_sub)
        handler.start()

        try:
            # Test routing to root model
            llm_query_root = handler.create_llm_query_fn("root")
            root_response = llm_query_root("Test root query")
            self.assertEqual(root_response, "Root model response")

            # Test routing to sub model
            llm_query_sub = handler.create_llm_query_fn("sub")
            sub_response = llm_query_sub("Test sub query")
            self.assertEqual(sub_response, "Sub model response")

            # Verify usage tracking is independent
            all_usage = handler.get_all_usage()
            self.assertEqual(all_usage["root"]["total_tokens"], 300)
            self.assertEqual(all_usage["sub"]["total_tokens"], 30)
        finally:
            handler.stop()

    def test_repl_with_sub_model_routing(self):
        """E2E: LocalREPL routes llm_query() to sub-model for cost optimization."""
        # Create mock sub-model
        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        async def mock_completion(messages):
            response = MagicMock()
            response.content = "Sub model answer: 4"
            return response

        mock_lm_sub.side_effect = mock_completion

        # Create client and handler
        client_sub = SynalinksLMClient(mock_lm_sub)
        handler = LMHandler()
        handler.register_client("sub", client_sub)
        handler.start()

        try:
            # Create REPL with sub-model routing
            llm_query_fn = handler.create_llm_query_fn("sub")
            repl = LocalREPL(llm_query_fn=llm_query_fn, default_sub_model="sub")

            # Execute code that uses llm_query
            result = repl.execute("answer = llm_query('What is 2+2?')")

            # Verify execution succeeded
            self.assertTrue(result.success)
            self.assertIn("answer", result.locals)
            self.assertEqual(result.locals["answer"], "Sub model answer: 4")

            # Verify default_sub_model is set
            self.assertEqual(repl.default_sub_model, "sub")
        finally:
            handler.stop()

    def test_recursive_generator_with_multi_model_setup(self):
        """E2E: RecursiveGenerator with separate root and sub models."""
        # Create mock LMs
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"

        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        # Create RecursiveGenerator with both models
        gen = RecursiveGenerator(
            language_model=mock_lm_root,
            sub_language_model=mock_lm_sub,
            max_iterations=10,
        )

        # Verify both models are set correctly
        self.assertEqual(gen.language_model.model, "zai/glm-4.7")
        self.assertEqual(gen.sub_language_model.model, "groq/openai/gpt-oss-20b")
        self.assertIsNot(gen.language_model, gen.sub_language_model)

    def test_full_stack_multi_model_integration(self):
        """E2E: Full stack integration - RecursiveGenerator, LMHandler, clients."""
        # Create mock LMs with cost difference
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"

        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        # Setup completion mocks with different costs
        async def mock_root_completion(messages):
            response = MagicMock()
            response.content = "Expensive root orchestration"
            response.usage = {
                "prompt_tokens": 500,
                "completion_tokens": 1000,
                "total_tokens": 1500,
            }
            return response

        async def mock_sub_completion(messages):
            response = MagicMock()
            response.content = "Cheap sub-task execution"
            response.usage = {
                "prompt_tokens": 20,
                "completion_tokens": 40,
                "total_tokens": 60,
            }
            return response

        mock_lm_root.side_effect = mock_root_completion
        mock_lm_sub.side_effect = mock_sub_completion

        # Create full stack
        client_root = SynalinksLMClient(mock_lm_root)
        client_sub = SynalinksLMClient(mock_lm_sub)

        handler = LMHandler()
        handler.register_client("root", client_root)
        handler.register_client("sub", client_sub)
        handler.start()

        try:
            # Create RecursiveGenerator with both models
            gen = RecursiveGenerator(
                language_model=mock_lm_root,
                sub_language_model=mock_lm_sub,
                max_iterations=5,
            )

            # Verify models are distinct
            self.assertIsNot(gen.language_model, gen.sub_language_model)

            # Simulate calls to both models
            llm_query_root = handler.create_llm_query_fn("root")
            llm_query_sub = handler.create_llm_query_fn("sub")

            root_resp = llm_query_root("Orchestrate task")
            sub_resp = llm_query_sub("Execute subtask")

            # Verify responses
            self.assertEqual(root_resp, "Expensive root orchestration")
            self.assertEqual(sub_resp, "Cheap sub-task execution")

            # Verify usage shows cost difference
            all_usage = handler.get_all_usage()
            root_usage = all_usage["root"]
            sub_usage = all_usage["sub"]

            # Root should be more expensive
            self.assertEqual(root_usage["total_tokens"], 1500)
            self.assertEqual(sub_usage["total_tokens"], 60)
            self.assertGreater(root_usage["total_tokens"], sub_usage["total_tokens"] * 10)

            # Verify clients can be retrieved
            self.assertEqual(handler.get_client("root"), client_root)
            self.assertEqual(handler.get_client("sub"), client_sub)
        finally:
            handler.stop()

    def test_serialization_preserves_multi_model_setup(self):
        """E2E: Serialization/deserialization preserves both models."""
        # Create mock LMs
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"

        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        # Create RecursiveGenerator
        gen1 = RecursiveGenerator(
            language_model=mock_lm_root,
            sub_language_model=mock_lm_sub,
            max_iterations=20,
            name="multi_model_gen",
        )

        # Serialize
        config = gen1.get_config()

        # Verify config has both models
        self.assertIn("language_model", config)
        self.assertIn("sub_language_model", config)
        self.assertEqual(config["language_model"].model, "zai/glm-4.7")
        self.assertEqual(config["sub_language_model"].model, "groq/openai/gpt-oss-20b")

        # Deserialize
        gen2 = RecursiveGenerator.from_config(config)

        # Verify both models restored
        self.assertEqual(gen2.language_model.model, "zai/glm-4.7")
        self.assertEqual(gen2.sub_language_model.model, "groq/openai/gpt-oss-20b")
        self.assertEqual(gen2.max_iterations, 20)
        self.assertEqual(gen2.name, "multi_model_gen")

    def test_default_sub_model_to_language_model_e2e(self):
        """E2E: When sub_language_model not provided, it defaults to language_model."""
        # Create single mock LM
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        # Create RecursiveGenerator with only language_model
        gen = RecursiveGenerator(language_model=mock_lm, max_iterations=10)

        # Verify sub_language_model defaults to language_model
        self.assertEqual(gen.language_model, mock_lm)
        self.assertEqual(gen.sub_language_model, mock_lm)
        self.assertIs(gen.sub_language_model, gen.language_model)

        # Verify both point to same model
        self.assertEqual(gen.language_model.model, "zai/glm-4.7")
        self.assertEqual(gen.sub_language_model.model, "zai/glm-4.7")

        # Create client
        client = SynalinksLMClient(mock_lm)

        # Create handler with single client
        handler = LMHandler()
        handler.register_client("default", client)

        # Verify single client can handle both root and sub queries
        root_client = handler.get_client("default")
        self.assertIsNotNone(root_client)
        self.assertEqual(root_client.model_name, "zai/glm-4.7")
