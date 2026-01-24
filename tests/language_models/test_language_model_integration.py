# License Apache 2.0: (c) 2025 Synalinks Team

"""Integration tests for LanguageModel structured output with real providers.

Run with:
    uv run --env-file .env -- python -m pytest tests/language_models/test_language_model_integration.py -v --override-ini="addopts="
"""

import os

import pytest

from synalinks.src.backend import ChatMessage, ChatMessages, ChatRole
from synalinks.src.language_models.language_model import LanguageModel


@pytest.mark.asyncio
async def test_groq_structured_output_with_regex_backslash():
    api_key = os.environ.get("GROQ_API_KEY")
    assert api_key, "GROQ_API_KEY is required for integration tests"

    lm = LanguageModel(model="groq/moonshotai/kimi-k2-instruct-0905")
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "code_lines": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "code_lines"],
    }
    messages = ChatMessages(
        messages=[
            ChatMessage(
                role=ChatRole.USER,
                content=(
                    "Return a JSON object with reasoning and code_lines."
                    " The code_lines must include a regex string like r'^\\s*\\w+'."
                ),
            )
        ]
    )

    result = await lm(messages, schema=schema)
    assert isinstance(result, dict)
    assert "reasoning" in result
    assert "code_lines" in result
    assert isinstance(result["code_lines"], list)


@pytest.mark.asyncio
async def test_zai_structured_output_minimal():
    api_key = os.environ.get("ZAI_API_KEY")
    assert api_key, "ZAI_API_KEY is required for integration tests"

    lm = LanguageModel(
        model="openai/glm-4.7",
        api_base="https://api.z.ai/api/coding/paas/v4",
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    messages = ChatMessages(
        messages=[
            ChatMessage(
                role=ChatRole.USER,
                content="Return a JSON object with value='ok'.",
            )
        ]
    )

    result = await lm(messages, schema=schema)
    assert isinstance(result, dict)
    assert result.get("value")
