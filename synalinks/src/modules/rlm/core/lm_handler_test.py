"""Tests for LMHandler."""

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
