# License Apache 2.0: (c) 2025 Synalinks Team

import json

import pytest
import litellm

from synalinks.src.backend import ChatMessage, ChatMessages, ChatRole, DataModel
from synalinks.src.language_models.language_model import LanguageModel


def _make_groq_error(failed_payload: str) -> Exception:
    wrapper = {"error": {"failed_generation": failed_payload}}
    message = f"GroqException - {json.dumps(wrapper)}"
    return RuntimeError(message)


def test_recover_groq_wraps_missing_code_lines():
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "code_lines": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "code_lines"],
    }
    failed = json.dumps({"answer": "ok", "score": 1})
    error = _make_groq_error(failed)

    recovered = LanguageModel._recover_groq_failed_generation(schema, error)

    assert recovered is not None
    assert recovered["reasoning"] == ""
    assert recovered["code_lines"] == ["SUBMIT(answer='ok', score=1)"]


def test_recover_groq_fills_missing_reasoning():
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "code_lines": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "code_lines"],
    }
    failed = json.dumps({"code_lines": ["print('ready')"]})
    error = _make_groq_error(failed)

    recovered = LanguageModel._recover_groq_failed_generation(schema, error)

    assert recovered is not None
    assert recovered["reasoning"] == ""
    assert recovered["code_lines"] == ["print('ready')"]


def test_recover_groq_returns_none_without_failed_generation():
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "code_lines": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "code_lines"],
    }
    error = RuntimeError("GroqException - something else")

    recovered = LanguageModel._recover_groq_failed_generation(schema, error)

    assert recovered is None


def test_recover_groq_fallback_on_parse_failure():
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "code_lines": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "code_lines"],
    }
    failed = "{not json"
    error = _make_groq_error(failed)

    recovered = LanguageModel._recover_groq_failed_generation(schema, error)

    assert recovered == {"reasoning": "", "code_lines": ["print('retry')"]}


def test_parse_json_candidate_strips_fences_and_trailing_text():
    text = "```json\n{\"a\": 1}\n```\nExtra text"
    parsed = LanguageModel._parse_json_candidate(text)
    assert parsed == {"a": 1}


def test_parse_json_candidate_handles_invalid_backslash_escape():
    text = '{"code_lines":["re.findall(r\'^\\s*\\\\w+\', text)"]}'
    parsed = LanguageModel._parse_json_candidate(text)
    assert isinstance(parsed, dict)
    assert "code_lines" in parsed
    assert isinstance(parsed["code_lines"], list)


def test_finalize_structured_output_coerces_list_and_string_types():
    schema = {
        "type": "object",
        "properties": {
            "code_lines": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["code_lines", "summary"],
    }
    value = {"code_lines": "print('ok')", "summary": ["a", "b"]}
    normalized, error = LanguageModel._finalize_structured_output(schema, value)
    assert error is None
    assert normalized["code_lines"] == ["print('ok')"]
    assert normalized["summary"] == "a\nb"


def test_finalize_structured_output_drops_unknown_keys():
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }
    value = {"summary": "ok", "extra": 123}
    normalized, error = LanguageModel._finalize_structured_output(schema, value)
    assert error is None
    assert normalized == {"summary": "ok"}


def test_validate_with_data_model_success():
    class Result(DataModel):
        value: int

    validated, error = LanguageModel._validate_with_data_model(Result, {"value": "2"})
    assert error is None
    assert validated["value"] == 2


def test_validate_with_data_model_failure():
    class Result(DataModel):
        value: int

    validated, error = LanguageModel._validate_with_data_model(Result, {"value": "nope"})
    assert validated is None
    assert error is not None


@pytest.mark.asyncio
async def test_data_model_validation_triggers_repair(monkeypatch):
    class Result(DataModel):
        value: int

    async def fake_acompletion(**kwargs):
        class DummyResponse(dict):
            _hidden_params = {"response_cost": 0.0}

        return DummyResponse(
            {
                "choices": [
                    {"message": {"content": "{\"value\": \"not-an-int\"}"}}
                ]
            }
        )

    async def fake_repair(self, schema, invalid_output, error_summary, base_kwargs):
        return {"value": 1}

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(LanguageModel, "_repair_structured_output", fake_repair)

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    messages = ChatMessages(
        messages=[ChatMessage(role=ChatRole.USER, content="Return JSON.")]
    )
    lm = LanguageModel(model="openai/gpt-4o-mini", retry=1)
    result = await lm(messages, schema=schema, data_model=Result)

    assert result == {"value": 1}
