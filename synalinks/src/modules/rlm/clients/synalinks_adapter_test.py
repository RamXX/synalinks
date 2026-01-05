"""Tests for SynalinksLMClient adapter."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from synalinks.src import testing
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.modules.rlm.clients.synalinks_adapter import SynalinksLMClient


class SynalinksLMClientTest(testing.TestCase):
    """Tests for SynalinksLMClient adapter."""

    def test_model_name_exposure(self):
        """Model name is exposed from wrapped LanguageModel."""
        mock_lm = MagicMock()
        mock_lm.model = "zai/glm-4.7"

        client = SynalinksLMClient(mock_lm)

        self.assertEqual(client.model_name, "zai/glm-4.7")

    def test_multi_client_setup_different_models(self):
        """Can create multiple clients with different models."""
        mock_lm_root = MagicMock()
        mock_lm_root.model = "zai/glm-4.7"

        mock_lm_sub1 = MagicMock()
        mock_lm_sub1.model = "groq/openai/gpt-oss-20b"

        mock_lm_sub2 = MagicMock()
        mock_lm_sub2.model = "openai/gpt-4"

        client_root = SynalinksLMClient(mock_lm_root)
        client_sub1 = SynalinksLMClient(mock_lm_sub1)
        client_sub2 = SynalinksLMClient(mock_lm_sub2)

        self.assertEqual(client_root.model_name, "zai/glm-4.7")
        self.assertEqual(client_sub1.model_name, "groq/openai/gpt-oss-20b")
        self.assertEqual(client_sub2.model_name, "openai/gpt-4")

    def test_string_prompt_conversion(self):
        """String prompt converts to ChatMessages."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        client = SynalinksLMClient(mock_lm)
        result = client._convert_prompt("Hello, world!")

        self.assertIsInstance(result, ChatMessages)
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].role, ChatRole.USER)
        self.assertEqual(result.messages[0].content, "Hello, world!")

    def test_list_prompt_conversion(self):
        """List prompt converts to ChatMessages."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        client = SynalinksLMClient(mock_lm)
        prompt = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Explain more."},
        ]
        result = client._convert_prompt(prompt)

        self.assertIsInstance(result, ChatMessages)
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.messages[0].role, "user")
        self.assertEqual(result.messages[0].content, "What is AI?")
        self.assertEqual(result.messages[1].role, "assistant")
        self.assertEqual(result.messages[2].role, "user")

    def test_sync_completion_runs_async_in_event_loop(self):
        """Sync completion() method runs async LM in event loop."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        # Create async mock that returns a response
        async def mock_call(messages):
            response = MagicMock()
            response.content = "Test response"
            response.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
            return response

        mock_lm.side_effect = mock_call

        client = SynalinksLMClient(mock_lm)
        result = client.completion("Test prompt")

        self.assertEqual(result, "Test response")
        mock_lm.assert_called_once()

    async def test_async_acompletion_method(self):
        """Async acompletion() method works correctly."""
        mock_lm = AsyncMock()
        mock_lm.model = "test-model"

        # Mock the async call
        mock_response = MagicMock()
        mock_response.content = "Async test response"
        mock_response.usage = {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        }
        mock_lm.return_value = mock_response

        client = SynalinksLMClient(mock_lm)
        result = await client.acompletion("Test async prompt")

        self.assertEqual(result, "Async test response")
        mock_lm.assert_awaited_once()

    def test_usage_tracking_per_client_independence(self):
        """Usage tracking is independent per client."""
        mock_lm_root = MagicMock()
        mock_lm_root.model = "expensive-model"

        mock_lm_sub = MagicMock()
        mock_lm_sub.model = "cheap-model"

        # Setup async mocks for both
        async def mock_root_call(messages):
            response = MagicMock()
            response.content = "Root response"
            response.usage = {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
            }
            return response

        async def mock_sub_call(messages):
            response = MagicMock()
            response.content = "Sub response"
            response.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
            return response

        mock_lm_root.side_effect = mock_root_call
        mock_lm_sub.side_effect = mock_sub_call

        client_root = SynalinksLMClient(mock_lm_root)
        client_sub = SynalinksLMClient(mock_lm_sub)

        # Make calls to each client
        client_root.completion("Root query")
        client_sub.completion("Sub query")

        # Check independent usage tracking
        root_usage = client_root.get_usage_summary()
        sub_usage = client_sub.get_usage_summary()

        self.assertEqual(root_usage["prompt_tokens"], 100)
        self.assertEqual(root_usage["completion_tokens"], 200)
        self.assertEqual(root_usage["total_tokens"], 300)
        self.assertEqual(root_usage["calls"], 1)

        self.assertEqual(sub_usage["prompt_tokens"], 10)
        self.assertEqual(sub_usage["completion_tokens"], 20)
        self.assertEqual(sub_usage["total_tokens"], 30)
        self.assertEqual(sub_usage["calls"], 1)

    def test_usage_reset_functionality(self):
        """Usage reset clears counters correctly."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        async def mock_call(messages):
            response = MagicMock()
            response.content = "Response"
            response.usage = {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            }
            return response

        mock_lm.side_effect = mock_call

        client = SynalinksLMClient(mock_lm)

        # Make a call
        client.completion("First query")
        usage_before = client.get_usage_summary()
        self.assertEqual(usage_before["total_tokens"], 150)
        self.assertEqual(usage_before["calls"], 1)

        # Reset usage
        client.reset_usage()
        usage_after = client.get_usage_summary()

        self.assertEqual(usage_after["prompt_tokens"], 0)
        self.assertEqual(usage_after["completion_tokens"], 0)
        self.assertEqual(usage_after["total_tokens"], 0)
        self.assertEqual(usage_after["calls"], 0)

    def test_response_extraction_from_dict_format(self):
        """Response extraction works with dict format."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        client = SynalinksLMClient(mock_lm)

        # Test dict with content field
        result_dict = {"content": "Dict response"}
        extracted = client._extract_response(result_dict)
        self.assertEqual(extracted, "Dict response")

    def test_response_extraction_from_object_format(self):
        """Response extraction works with object format."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        client = SynalinksLMClient(mock_lm)

        # Test object with content attribute
        result_obj = MagicMock()
        result_obj.content = "Object response"
        extracted = client._extract_response(result_obj)
        self.assertEqual(extracted, "Object response")

    def test_response_extraction_from_string_format(self):
        """Response extraction works with string format."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        client = SynalinksLMClient(mock_lm)

        # Test plain string
        result_str = "Plain string response"
        extracted = client._extract_response(result_str)
        self.assertEqual(extracted, "Plain string response")

    def test_usage_accumulation_across_multiple_calls(self):
        """Usage accumulates correctly across multiple calls."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        call_count = 0

        async def mock_call(messages):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.content = f"Response {call_count}"
            response.usage = {
                "prompt_tokens": 10 * call_count,
                "completion_tokens": 20 * call_count,
                "total_tokens": 30 * call_count,
            }
            return response

        mock_lm.side_effect = mock_call

        client = SynalinksLMClient(mock_lm)

        # Make 3 calls
        client.completion("Query 1")
        client.completion("Query 2")
        client.completion("Query 3")

        usage = client.get_usage_summary()

        # Should accumulate: (10+20+30) + (20+40+60) + (30+60+90)
        self.assertEqual(usage["prompt_tokens"], 60)  # 10+20+30
        self.assertEqual(usage["completion_tokens"], 120)  # 20+40+60
        self.assertEqual(usage["total_tokens"], 180)  # 30+60+90
        self.assertEqual(usage["calls"], 3)

    def test_last_usage_tracking(self):
        """Last usage tracks most recent call separately."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        call_count = 0

        async def mock_call(messages):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.content = f"Response {call_count}"
            response.usage = {
                "prompt_tokens": 10 * call_count,
                "completion_tokens": 20 * call_count,
                "total_tokens": 30 * call_count,
            }
            return response

        mock_lm.side_effect = mock_call

        client = SynalinksLMClient(mock_lm)

        # Make first call
        client.completion("Query 1")
        last_usage_1 = client.get_last_usage()
        self.assertEqual(last_usage_1["prompt_tokens"], 10)

        # Make second call
        client.completion("Query 2")
        last_usage_2 = client.get_last_usage()
        self.assertEqual(last_usage_2["prompt_tokens"], 20)
        self.assertEqual(last_usage_2["completion_tokens"], 40)

    def test_unsupported_prompt_type_raises_error(self):
        """Unsupported prompt type raises ValueError."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        client = SynalinksLMClient(mock_lm)

        with self.assertRaises(ValueError) as ctx:
            client._convert_prompt(12345)  # Invalid type

        self.assertIn("Unsupported prompt type", str(ctx.exception))

    def test_last_usage_property(self):
        """last_usage property returns most recent call usage."""
        mock_lm = MagicMock()
        mock_lm.model = "test-model"

        async def mock_call(messages):
            response = MagicMock()
            response.content = "Response"
            response.usage = {
                "prompt_tokens": 42,
                "completion_tokens": 84,
                "total_tokens": 126,
            }
            return response

        mock_lm.side_effect = mock_call

        client = SynalinksLMClient(mock_lm)

        # Make a call
        client.completion("Test query")

        # Access via property
        last = client.last_usage
        self.assertEqual(last["prompt_tokens"], 42)
        self.assertEqual(last["completion_tokens"], 84)
        self.assertEqual(last["total_tokens"], 126)
