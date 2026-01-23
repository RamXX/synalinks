# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel


class LanguageModelTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_call_api_without_structured_output(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello, how can I help you?"}}]
        }

        expected = ChatMessage(
            role=ChatRole.ASSISTANT, content="Hello, how can I help you?"
        )
        result = await language_model(messages)
        self.assertEqual(result, ChatMessage(**result).get_json())
        self.assertEqual(result, expected.get_json())

    @patch("litellm.acompletion")
    async def test_call_api_with_structured_output(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")

        messages = ChatMessages(
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    content="What is the french city of aerospace and robotics?",
                )
            ]
        )

        expected_string = (
            """{"rationale": "Toulouse hosts numerous research institutions """
            """and universities that specialize in aerospace engineering and """
            """robotics, such as the Institut Supérieur de l'Aéronautique et """
            """de l'Espace (ISAE-SUPAERO) and the French National Centre for """
            """Scientific Research (CNRS)","""
            """ "answer": "Toulouse"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        expected = AnswerWithRationale(
            rationale=(
                "Toulouse hosts numerous research institutions and universities "
                "that specialize in aerospace engineering and robotics, such as "
                "the Institut Supérieur de l'Aéronautique et de l'Espace "
                "(ISAE-SUPAERO) and the "
                "French National Centre for Scientific Research (CNRS)"
            ),
            answer="Toulouse",
        )
        result = await language_model(messages, schema=AnswerWithRationale.get_schema())
        self.assertEqual(result, AnswerWithRationale(**result).get_json())
        self.assertEqual(result, expected.get_json())

    @patch("litellm.acompletion")
    async def test_call_api_streaming_mode(self, mock_completion):
        language_model = LanguageModel(model="ollama/deepseek-r1")

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        mock_response_iterator = iter(
            [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo,"}}]},
                {"choices": [{"delta": {"content": " how"}}]},
                {"choices": [{"delta": {"content": " can"}}]},
                {"choices": [{"delta": {"content": " I"}}]},
                {"choices": [{"delta": {"content": " help"}}]},
                {"choices": [{"delta": {"content": " you?"}}]},
            ]
        )

        mock_completion.return_value = mock_response_iterator

        expected = "Hello, how can I help you?"

        response = await language_model(messages, streaming=True)

        result = ""
        for msg in response:
            result += msg.get("content")

        self.assertEqual(result, expected)

    def test_clean_messages_for_groq_strips_system_user_fields(self):
        """Test that system and user messages have tool fields stripped."""
        messages = [
            {
                "role": "system",
                "content": "You are helpful.",
                "tool_call_id": None,
                "tool_calls": [],
            },
            {
                "role": "user",
                "content": "Hello",
                "tool_call_id": None,
                "tool_calls": [],
            },
        ]

        cleaned = LanguageModel._clean_messages_for_groq(messages)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0], {"role": "system", "content": "You are helpful."})
        self.assertEqual(cleaned[1], {"role": "user", "content": "Hello"})
        self.assertNotIn("tool_calls", cleaned[0])
        self.assertNotIn("tool_call_id", cleaned[0])
        self.assertNotIn("tool_calls", cleaned[1])
        self.assertNotIn("tool_call_id", cleaned[1])

    def test_clean_messages_for_groq_preserves_assistant_tool_calls(self):
        """Test that assistant messages retain tool_calls when present."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call],
            },
        ]

        cleaned = LanguageModel._clean_messages_for_groq(messages)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["role"], "assistant")
        self.assertEqual(cleaned[0]["tool_calls"], [tool_call])

    def test_clean_messages_for_groq_strips_empty_assistant_tool_calls(self):
        """Test that assistant messages without tool_calls don't include the field."""
        messages = [
            {
                "role": "assistant",
                "content": "Hello!",
                "tool_calls": [],
            },
        ]

        cleaned = LanguageModel._clean_messages_for_groq(messages)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0], {"role": "assistant", "content": "Hello!"})
        self.assertNotIn("tool_calls", cleaned[0])

    def test_clean_messages_for_groq_preserves_tool_messages(self):
        """Test that tool messages retain tool_call_id and name."""
        messages = [
            {
                "role": "tool",
                "content": '{"temperature": 22}',
                "tool_call_id": "call_123",
                "name": "get_weather",
            },
        ]

        cleaned = LanguageModel._clean_messages_for_groq(messages)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["role"], "tool")
        self.assertEqual(cleaned[0]["content"], '{"temperature": 22}')
        self.assertEqual(cleaned[0]["tool_call_id"], "call_123")
        self.assertEqual(cleaned[0]["name"], "get_weather")

    def test_enforce_no_additional_properties_recursive(self):
        """Test recursive additionalProperties enforcement."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                    },
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                        },
                    },
                },
                "choice": {
                    "oneOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}},
                        {"type": "object", "properties": {"b": {"type": "string"}}},
                    ]
                },
            },
        }

        hardened = LanguageModel._enforce_no_additional_properties(schema)

        assert hardened["additionalProperties"] is False
        assert hardened["properties"]["nested"]["additionalProperties"] is False
        assert (
            hardened["properties"]["items"]["items"]["additionalProperties"] is False
        )
        assert hardened["properties"]["choice"]["oneOf"][0]["additionalProperties"] is False
        assert hardened["properties"]["choice"]["oneOf"][1]["additionalProperties"] is False

    @patch("litellm.acompletion")
    async def test_groq_model_cleans_messages(self, mock_completion):
        """Test that Groq models have messages cleaned before API call."""
        language_model = LanguageModel(model="groq/llama-3.3-70b-versatile")

        messages = ChatMessages(
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="You are helpful."),
                ChatMessage(role=ChatRole.USER, content="Hello"),
            ]
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hi there!"}}]
        }

        await language_model(messages)

        # Verify the messages passed to litellm are clean
        call_args = mock_completion.call_args
        sent_messages = call_args.kwargs.get("messages") or call_args[1].get("messages")

        for msg in sent_messages:
            if msg["role"] in ("system", "user"):
                self.assertNotIn("tool_calls", msg)
                self.assertNotIn("tool_call_id", msg)
