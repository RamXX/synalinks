"""Tests for LMHandler."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from synalinks.src import testing
from synalinks.src.modules.rlm.clients.synalinks_adapter import SynalinksLMClient
from synalinks.src.modules.rlm.core.lm_handler import LMHandler


class LMHandlerTest(testing.TestCase):
    """Tests for LMHandler TCP socket routing."""

    def test_instantiates_without_error(self):
        """LMHandler instantiates successfully."""
        handler = LMHandler()
        self.assertIsNotNone(handler)

    def test_register_client(self):
        """Can register LM clients."""
        handler = LMHandler()
        mock_lm = MagicMock()
        mock_lm.model = "test"
        client = SynalinksLMClient(mock_lm)

        handler.register_client("root", client)
        self.assertIn("root", handler._clients)

    def test_multi_client_registration(self):
        """Can register multiple clients."""
        handler = LMHandler()

        mock_lm_root = MagicMock()
        mock_lm_root.model = "zai/glm-4.7"
        client_root = SynalinksLMClient(mock_lm_root)

        mock_lm_sub = MagicMock()
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"
        client_sub = SynalinksLMClient(mock_lm_sub)

        handler.register_client("root", client_root)
        handler.register_client("sub", client_sub)

        self.assertEqual(len(handler._clients), 2)
        self.assertIn("root", handler._clients)
        self.assertIn("sub", handler._clients)

    def test_start_stop(self):
        """Handler can start and stop server."""
        handler = LMHandler()
        handler.start()
        self.assertTrue(handler._running)
        self.assertIsNotNone(handler.get_port())

        handler.stop()
        self.assertFalse(handler._running)

    def test_creates_llm_query_function(self):
        """Can create llm_query function for REPL."""
        handler = LMHandler()
        mock_lm = MagicMock()
        mock_lm.model = "test"
        client = SynalinksLMClient(mock_lm)
        handler.register_client("test", client)

        llm_query_fn = handler.create_llm_query_fn("test")
        self.assertIsNotNone(llm_query_fn)
        self.assertTrue(callable(llm_query_fn))

    def test_default_client(self):
        """First registered client becomes default."""
        handler = LMHandler()

        mock_lm1 = MagicMock()
        mock_lm1.model = "model1"
        client1 = SynalinksLMClient(mock_lm1)

        mock_lm2 = MagicMock()
        mock_lm2.model = "model2"
        client2 = SynalinksLMClient(mock_lm2)

        handler.register_client("first", client1)
        handler.register_client("second", client2)

        self.assertEqual(handler._default_client, client1)

    def test_get_client_returns_registered_client(self):
        """get_client() returns correct client by name."""
        handler = LMHandler()

        mock_lm_root = MagicMock()
        mock_lm_root.model = "zai/glm-4.7"
        client_root = SynalinksLMClient(mock_lm_root)

        mock_lm_sub = MagicMock()
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"
        client_sub = SynalinksLMClient(mock_lm_sub)

        handler.register_client("root", client_root)
        handler.register_client("sub", client_sub)

        # Get by name
        retrieved_root = handler.get_client("root")
        retrieved_sub = handler.get_client("sub")

        self.assertEqual(retrieved_root, client_root)
        self.assertEqual(retrieved_sub, client_sub)
        self.assertEqual(retrieved_root.model_name, "zai/glm-4.7")
        self.assertEqual(retrieved_sub.model_name, "groq/openai/gpt-oss-20b")

    def test_get_client_returns_none_for_missing_client(self):
        """get_client() returns None for unregistered name."""
        handler = LMHandler()

        mock_lm = MagicMock()
        mock_lm.model = "test"
        client = SynalinksLMClient(mock_lm)

        handler.register_client("exists", client)

        # Get existing client
        self.assertIsNotNone(handler.get_client("exists"))

        # Get non-existing client
        self.assertIsNone(handler.get_client("does_not_exist"))

    def test_get_all_usage_returns_empty_dict_when_no_clients(self):
        """get_all_usage() returns empty dict when no clients registered."""
        handler = LMHandler()
        usage = handler.get_all_usage()
        self.assertEqual(usage, {})

    def test_get_all_usage_returns_usage_from_all_clients(self):
        """get_all_usage() aggregates usage from multiple clients."""
        handler = LMHandler()

        mock_lm_root = MagicMock()
        mock_lm_root.model = "zai/glm-4.7"
        client_root = SynalinksLMClient(mock_lm_root)

        mock_lm_sub = MagicMock()
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"
        client_sub = SynalinksLMClient(mock_lm_sub)

        # Set up mock usage for root
        client_root._usage_total = {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "calls": 2,
        }

        # Set up mock usage for sub
        client_sub._usage_total = {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
            "calls": 5,
        }

        handler.register_client("root", client_root)
        handler.register_client("sub", client_sub)

        usage = handler.get_all_usage()

        # Verify structure
        self.assertIn("root", usage)
        self.assertIn("sub", usage)

        # Verify root usage
        self.assertEqual(usage["root"]["prompt_tokens"], 100)
        self.assertEqual(usage["root"]["completion_tokens"], 200)
        self.assertEqual(usage["root"]["total_tokens"], 300)
        self.assertEqual(usage["root"]["calls"], 2)

        # Verify sub usage
        self.assertEqual(usage["sub"]["prompt_tokens"], 50)
        self.assertEqual(usage["sub"]["completion_tokens"], 100)
        self.assertEqual(usage["sub"]["total_tokens"], 150)
        self.assertEqual(usage["sub"]["calls"], 5)

    async def test_acompletion_batched_executes_concurrently(self):
        """acompletion_batched executes prompts concurrently."""
        handler = LMHandler()

        # Create a mock acompletion that returns different responses
        async def mock_acompletion(prompt):
            return f"Response to: {prompt}"

        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(side_effect=mock_acompletion)

        handler.register_client("test", mock_client)

        prompts = ["Q1", "Q2", "Q3"]
        results = await handler.acompletion_batched(prompts, "test")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "Response to: Q1")
        self.assertEqual(results[1], "Response to: Q2")
        self.assertEqual(results[2], "Response to: Q3")

    async def test_acompletion_batched_partial_errors(self):
        """acompletion_batched handles individual errors without failing entire batch."""
        handler = LMHandler()

        # Create a mock that fails on second prompt
        call_count = [0]

        async def mock_acompletion(prompt):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Simulated error on Q2")
            return f"Response to: {prompt}"

        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(side_effect=mock_acompletion)

        handler.register_client("test", mock_client)

        prompts = ["Q1", "Q2", "Q3"]
        results = await handler.acompletion_batched(prompts, "test")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "Response to: Q1")
        self.assertIsInstance(results[1], ValueError)
        self.assertEqual(str(results[1]), "Simulated error on Q2")
        self.assertEqual(results[2], "Response to: Q3")

    async def test_acompletion_batched_no_client_raises(self):
        """acompletion_batched raises if no client registered."""
        handler = LMHandler()

        with self.assertRaises(RuntimeError) as ctx:
            await handler.acompletion_batched(["test"])

        self.assertIn("No client registered", str(ctx.exception))

    def test_create_llm_query_batched_fn(self):
        """Can create llm_query_batched function."""
        handler = LMHandler()
        mock_lm = MagicMock()
        mock_lm.model = "test"
        client = SynalinksLMClient(mock_lm)
        handler.register_client("test", client)

        llm_query_batched_fn = handler.create_llm_query_batched_fn("test")
        self.assertIsNotNone(llm_query_batched_fn)
        self.assertTrue(callable(llm_query_batched_fn))
