# License Apache 2.0: (c) 2025 Synalinks Team

"""Tests for Generator module."""

import pytest

from synalinks.src.backend import JsonDataModel
from synalinks.src.modules.core.generator import Generator


class RecordingLanguageModel:
    """Record kwargs passed to LanguageModel."""

    def __init__(self):
        self.last_kwargs = None

    async def __call__(self, messages, schema=None, streaming=False, **kwargs):
        self.last_kwargs = kwargs
        return {"answer": "ok"}

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


@pytest.mark.asyncio
async def test_generator_forwards_max_tokens():
    """Generator should forward max_tokens to the language model call."""
    lm = RecordingLanguageModel()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    gen = Generator(schema=schema, language_model=lm, max_tokens=123)

    inputs = JsonDataModel(
        json={"query": "hi"},
        schema={"type": "object", "properties": {"query": {"type": "string"}}},
        name="inputs",
    )

    result = await gen(inputs)

    assert result is not None
    assert lm.last_kwargs is not None
    assert lm.last_kwargs.get("max_tokens") == 123
