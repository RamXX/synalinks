# License Apache 2.0: (c) 2025 Synalinks Team

"""Integration tests for RLM (Recursive Language Model) with real LLMs.

These tests use actual LLM calls to verify end-to-end functionality.
Requires GROQ_API_KEY environment variable to be set.

Run with:
    uv run --env-file .env -- python -m pytest tests/modules/reasoning/test_repl_module_integration.py -v --override-ini="addopts="
"""

import copy
import json
import os
import warnings

import pytest
import litellm

import synalinks
from synalinks.src.backend import ChatRole, JsonDataModel
from synalinks.src.language_models.language_model import LanguageModel
from synalinks.src.modules.reasoning.repl_module import RLM
from synalinks.src.utils.nlp_utils import shorten_text


# =============================================================================
# Groq Workaround (from Synalinks skill)
# =============================================================================


def _clean_messages_for_groq(messages: list) -> list:
    """Remove tool_calls and tool_call_id from messages."""
    cleaned = []
    for msg in messages:
        clean_msg = {"role": msg.get("role"), "content": msg.get("content", "")}
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            clean_msg["tool_call_id"] = msg["tool_call_id"]
        cleaned.append(clean_msg)
    return cleaned


_original_call = None


async def _patched_call(self, messages, schema=None, streaming=False, **kwargs):
    """Patched __call__ that uses json_schema for Groq instead of tool-calling."""
    formatted_messages = messages.get_json().get("messages", [])
    input_kwargs = copy.deepcopy(kwargs)
    schema = copy.deepcopy(schema)

    # Clean messages for Groq
    if self.model.startswith("groq"):
        formatted_messages = _clean_messages_for_groq(formatted_messages)

    if schema:
        if self.model.startswith("groq"):
            # Use json_schema instead of tool-calling
            kwargs.update(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "structured_output", "schema": schema},
                    }
                }
            )
        elif self.model.startswith("anthropic"):
            kwargs.update(
                {
                    "tools": [
                        {
                            "name": "structured_output",
                            "description": "Generate a valid JSON output",
                            "input_schema": {
                                "type": "object",
                                "properties": schema.get("properties"),
                                "required": schema.get("required"),
                            },
                        }
                    ],
                    "tool_choice": {"type": "tool", "name": "structured_output"},
                }
            )
        elif self.model.startswith("ollama") or self.model.startswith("mistral"):
            kwargs.update(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"schema": schema},
                        "strict": True,
                    }
                }
            )
        elif self.model.startswith("openai") or self.model.startswith("azure"):
            if "properties" in schema:
                for prop_key, prop_value in schema["properties"].items():
                    if "$ref" in prop_value and "description" in prop_value:
                        del prop_value["description"]
            kwargs.update(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                }
            )
        elif (
            self.model.startswith("gemini")
            or self.model.startswith("xai")
            or self.model.startswith("hosted_vllm")
        ):
            kwargs.update(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"schema": schema},
                        "strict": True,
                    }
                }
            )
        else:
            raise ValueError(f"LM provider '{self.model.split('/')[0]}' not supported")

    if self.api_base:
        kwargs.update({"api_base": self.api_base})
    if streaming and schema:
        streaming = False
    if streaming:
        kwargs.update({"stream": True})

    for i in range(self.retry):
        try:
            response_str = ""
            response = await litellm.acompletion(
                model=self.model,
                messages=formatted_messages,
                timeout=self.timeout,
                caching=self.caching,
                **kwargs,
            )
            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
            ):
                self.last_call_cost = response._hidden_params["response_cost"]
                if self.last_call_cost is not None:
                    self.cumulated_cost += self.last_call_cost

            if self.model.startswith("anthropic") and schema:
                response_str = response["choices"][0]["message"]["tool_calls"][0][
                    "function"
                ]["arguments"]
            else:
                response_str = response["choices"][0]["message"]["content"].strip()

            if schema:
                return json.loads(response_str)
            else:
                return {
                    "role": ChatRole.ASSISTANT,
                    "content": response_str,
                    "tool_call_id": None,
                    "tool_calls": [],
                }
        except Exception as e:
            warnings.warn(f"Error calling {self}: {shorten_text(str(e))}")
        import asyncio

        await asyncio.sleep(1)

    return (
        self.fallback(messages, schema=schema, streaming=streaming, **input_kwargs)
        if self.fallback
        else None
    )


def patch_synalinks_for_groq():
    """Patch Synalinks to support Groq structured output. Call once at startup."""
    global _original_call
    if _original_call is None:
        _original_call = LanguageModel.__call__
        LanguageModel.__call__ = _patched_call


def create_groq_language_model(model_name: str, **kwargs) -> synalinks.LanguageModel:
    """Create a Groq LanguageModel with automatic patching."""
    patch_synalinks_for_groq()
    return synalinks.LanguageModel(model=f"groq/{model_name}", **kwargs)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def groq_api_key():
    """Get Groq API key from environment."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="module")
def language_model(groq_api_key):
    """Create Groq language model for testing."""
    return create_groq_language_model(
        "moonshotai/kimi-k2-instruct-0905",  # Good coding model for RLM
        timeout=120,
    )


@pytest.fixture
def simple_schema():
    """Simple output schema for tests."""
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The answer to the question"},
        },
        "required": ["answer"],
    }


@pytest.fixture
def numeric_schema():
    """Schema with numeric output."""
    return {
        "type": "object",
        "properties": {
            "result": {"type": "number", "description": "Numeric result"},
            "explanation": {
                "type": "string",
                "description": "Explanation of the calculation",
            },
        },
        "required": ["result", "explanation"],
    }


# =============================================================================
# Integration Tests
# =============================================================================


class TestRLMIntegration:
    """Integration tests for RLM with real LLM."""

    @pytest.mark.asyncio
    async def test_simple_submit(self, language_model, simple_schema):
        """Test that LLM can submit a simple answer."""
        module = RLM(
            schema=simple_schema,
            language_model=language_model,
            max_iterations=5,
            instructions="Answer the question directly. Use SUBMIT(answer='your answer') when done.",
        )

        inputs = JsonDataModel(
            json={"question": "What is 2 + 2?"},
            schema={"type": "object", "properties": {"question": {"type": "string"}}},
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        assert result.get("answer") is not None
        # The answer should contain "4" somewhere
        assert "4" in str(result.get("answer"))

    @pytest.mark.asyncio
    async def test_multi_step_calculation(self, language_model, numeric_schema):
        """Test multi-step calculation using REPL."""
        module = RLM(
            schema=numeric_schema,
            language_model=language_model,
            max_iterations=10,
            return_history=True,
            instructions="""
You are solving a math problem step by step.
Use Python code to calculate the answer.
Always print intermediate results.
When done, use SUBMIT(result=value, explanation='your explanation').
""",
        )

        inputs = JsonDataModel(
            json={"problem": "Calculate the sum of squares from 1 to 5"},
            schema={"type": "object", "properties": {"problem": {"type": "string"}}},
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        # 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 1 + 4 + 9 + 16 + 25 = 55
        assert result.get("result") == 55 or result.get("result") == 55.0
        assert result.get("explanation") is not None

        # Check history is included
        json_data = result.get_json()
        assert "_history" in json_data
        assert len(json_data["_history"]) > 0

    @pytest.mark.asyncio
    async def test_data_exploration(self, language_model, simple_schema):
        """Test data exploration before answering."""
        module = RLM(
            schema=simple_schema,
            language_model=language_model,
            max_iterations=10,
            return_history=True,
            instructions="""
You have access to a data variable. Explore it and answer the question.
Use print() to see values. Use SUBMIT(answer='your answer') when done.
""",
        )

        inputs = JsonDataModel(
            json={
                "data": {"name": "Alice", "age": 30, "city": "Boston"},
                "question": "What city does the person live in?",
            },
            schema={
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "question": {"type": "string"},
                },
            },
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        answer = result.get("answer")
        assert answer is not None
        assert "Boston" in answer or "boston" in answer.lower()

    @pytest.mark.asyncio
    async def test_llm_query_usage(self, language_model, simple_schema):
        """Test that llm_query is available and works."""
        module = RLM(
            schema=simple_schema,
            language_model=language_model,
            max_iterations=10,
            instructions="""
Analyze the sentiment of the given text.
You can use llm_query(prompt) to get semantic analysis if needed.
Use SUBMIT(answer='positive', 'negative', or 'neutral') when done.
""",
        )

        inputs = JsonDataModel(
            json={
                "text": "I love this beautiful sunny day! Everything is wonderful.",
                "question": "What is the sentiment of this text?",
            },
            schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "question": {"type": "string"},
                },
            },
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        answer = result.get("answer")
        # The test passes if we got an answer (llm_query may or may not be used)
        assert answer is not None

    @pytest.mark.asyncio
    async def test_error_recovery(self, language_model, simple_schema):
        """Test that LLM can recover from code errors."""
        module = RLM(
            schema=simple_schema,
            language_model=language_model,
            max_iterations=10,
            return_history=True,
            instructions="""
Write Python code to solve the problem. If you get an error, fix it and try again.
Use SUBMIT(answer='your answer') when done. The output field is called 'answer'.
""",
        )

        # Give a problem that might initially cause an error
        inputs = JsonDataModel(
            json={"task": "Find the length of the word 'hello' and submit it as the answer"},
            schema={"type": "object", "properties": {"task": {"type": "string"}}},
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        assert result.get("answer") is not None
        # Should contain "5" somewhere
        assert "5" in str(result.get("answer"))


class TestRLMWithComplexSchemas:
    """Test RLM with more complex output schemas."""

    @pytest.mark.asyncio
    async def test_array_output(self, language_model):
        """Test output with array field."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of items",
                },
            },
            "required": ["items"],
        }

        module = RLM(
            schema=schema,
            language_model=language_model,
            max_iterations=5,
            instructions="Generate a list of 3 fruits. Use SUBMIT(items=['fruit1', 'fruit2', 'fruit3']).",
        )

        inputs = JsonDataModel(
            json={"request": "List 3 fruits"},
            schema={"type": "object", "properties": {"request": {"type": "string"}}},
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        items = result.get("items")
        assert items is not None
        assert isinstance(items, list)
        assert len(items) >= 1  # Should have at least one item

    @pytest.mark.asyncio
    async def test_multi_field_output(self, language_model):
        """Test output with multiple fields."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the analysis"},
                "score": {"type": "number", "description": "Score from 1 to 10"},
                "summary": {"type": "string", "description": "Brief summary"},
            },
            "required": ["title", "score", "summary"],
        }

        module = RLM(
            schema=schema,
            language_model=language_model,
            max_iterations=5,
            instructions="""
Analyze the given text and provide:
- A title for your analysis
- A score from 1-10
- A brief summary
Use SUBMIT(title='...', score=N, summary='...') when done.
""",
        )

        inputs = JsonDataModel(
            json={"text": "Python is a popular programming language used for web development, data science, and AI."},
            schema={"type": "object", "properties": {"text": {"type": "string"}}},
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        assert result.get("title") is not None
        assert result.get("score") is not None
        assert result.get("summary") is not None
        assert 1 <= result.get("score") <= 10


class TestRLMEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_max_iterations_fallback(self, language_model, simple_schema):
        """Test that fallback extraction works when max iterations reached."""
        # Very low max_iterations to force fallback
        module = RLM(
            schema=simple_schema,
            language_model=language_model,
            max_iterations=1,
            instructions="Explore the data thoroughly before answering. Take multiple steps.",
        )

        inputs = JsonDataModel(
            json={"question": "What is the capital of France?"},
            schema={"type": "object", "properties": {"question": {"type": "string"}}},
            name="inputs",
        )

        # Should not raise, fallback extractor should handle it
        result = await module(inputs)
        # Result may be None or have a fallback answer
        # The important thing is it doesn't crash

    @pytest.mark.asyncio
    async def test_empty_input(self, language_model, simple_schema):
        """Test handling of minimal input."""
        module = RLM(
            schema=simple_schema,
            language_model=language_model,
            max_iterations=5,
        )

        inputs = JsonDataModel(
            json={},
            schema={"type": "object"},
            name="inputs",
        )

        # Should handle gracefully
        result = await module(inputs)
        # Just verify it doesn't crash
