# Provider-Specific Considerations

This document describes provider-specific behaviors and how Synalinks handles them internally.

---

## Overview

Synalinks uses [LiteLLM](https://github.com/BerriAI/litellm) as its abstraction layer for language model providers. While LiteLLM provides a unified interface, different providers have varying API requirements and behaviors. This document explains how Synalinks handles these differences.

---

## Groq

### Message Format Requirements

Groq's API enforces strict message schema validation. Unlike other providers (OpenAI, Anthropic, etc.) that silently ignore extra fields, Groq rejects messages containing fields not applicable to their role type.

#### Allowed Fields by Role

| Role | Allowed Fields |
|------|---------------|
| `system` | `role`, `content` |
| `user` | `role`, `content` |
| `assistant` | `role`, `content`, `tool_calls` (optional) |
| `tool` | `role`, `content`, `tool_call_id`, `name` |

#### The Problem

Synalinks' `ChatMessage` data model includes `tool_calls` and `tool_call_id` fields with default values (`[]` and `None`) on all messages. When serialized, these fields are present even when empty:

```json
{
  "role": "system",
  "content": "You are a helpful assistant.",
  "tool_call_id": null,
  "tool_calls": []
}
```

Groq's API rejects this with:
```
'messages.0' : for 'role:system' the following must be satisfied
[('messages.0' : property 'tool_calls' is unsupported)]
```

#### The Solution

The `LanguageModel` class automatically cleans messages when using Groq models. The internal `_clean_messages_for_groq()` method strips invalid fields from each message based on its role:

- **System/User messages**: Only `role` and `content` are retained
- **Assistant messages**: `tool_calls` is retained only if non-empty
- **Tool messages**: `tool_call_id` and `name` are retained

This cleaning happens transparently before making API calls.

### Structured Output Implementation

Groq supports native JSON schema response formats via `response_format`. When you provide a schema to a Groq model, Synalinks:

1. Adds `additionalProperties: false` to the schema (required by Groq)
2. Uses `response_format` with `type: json_schema`
3. Enables strict mode for schema validation

```python
# What Synalinks sends to Groq internally
{
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": {
                "type": "object",
                "properties": {...},
                "required": [...],
                "additionalProperties": false
            },
            "strict": true
        }
    }
}
```

**Important Implementation Details:**

1. **additionalProperties Requirement**: Groq's `json_schema` mode requires `additionalProperties: false` on object schemas. Synalinks automatically adds this to your schema.

2. **Native Schema Enforcement**: Unlike tool-based approaches, `json_schema` mode provides native schema validation, ensuring outputs always conform to the specified structure.

### Output Token Limits

For long structured outputs, set a max output token budget explicitly. Synalinks forwards `max_tokens` to the provider. You can set this per `LanguageModel` or per module (e.g., `Generator`, `RLM`).

```python
import synalinks

# Per-model cap
lm = synalinks.LanguageModel(
    model="groq/moonshotai/kimi-k2-instruct-0905",
    max_tokens=16384,  # match provider max output tokens
)

# Per-module cap
gen = synalinks.Generator(
    data_model=Answer,
    language_model=lm,
    max_tokens=4096,
)

rlm = synalinks.RLM(
    data_model=DeepAnalysis,
    language_model=lm,
    max_tokens=16384,
)
```

Refer to the Groq model docs for the current output-token limits for each model (e.g., Kimi and gpt-oss-20b).

### Supported Groq Models

All Groq-hosted models that support tool calling work with Synalinks:

- `groq/llama-3.3-70b-versatile`
- `groq/llama-3.1-8b-instant`
- `groq/llama-4-scout-17b-16e-instruct`
- `groq/qwen-qwq-32b`
- `groq/openai/gpt-oss-20b`
- `groq/openai/gpt-oss-120b`

### Example Usage

```python
import synalinks
import os

os.environ["GROQ_API_KEY"] = "your-api-key"

# Define your data models
class Query(synalinks.DataModel):
    question: str = synalinks.Field(description="The question to answer")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")
    confidence: float = synalinks.Field(description="Confidence score 0-1")

# Create the language model - any Groq model works
lm = synalinks.LanguageModel(model="groq/openai/gpt-oss-120b")

# Build a program
inputs = synalinks.Input(data_model=Query)
generator = synalinks.Generator(
    data_model=Answer,
    language_model=lm,
    instructions="Answer the question and provide a confidence score.",
)
outputs = await generator(inputs)
program = synalinks.Program(inputs=inputs, outputs=outputs)

# Run inference
result = await program(
    synalinks.JsonDataModel(
        schema=Query.get_schema(),
        json={"question": "What is the capital of France?"}
    )
)
print(result.get_json())
# Output: {"answer": "Paris", "confidence": 1.0}
```

---

## OpenAI / Azure OpenAI

OpenAI and Azure OpenAI use native JSON schema response formats with strict mode:

```python
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "structured_output",
        "strict": True,
        "schema": schema
    }
}
```

OpenAI disallows the `description` field in `$ref` schema definitions. Synalinks automatically removes these when present.

---

## Anthropic

Anthropic models use `response_format` with JSON schema. For newer models (Claude Sonnet 4.5, Opus 4.1), LiteLLM uses native output format. For older models, it falls back to tool-based structured output.

Anthropic models that support `reasoning_effort` expose their thinking via `reasoning_content` in the response. Synalinks automatically extracts this into the `thinking` field when using reasoning models.

---

## Ollama / Mistral

These providers use constrained structured output with JSON schema:

```python
response_format={
    "type": "json_schema",
    "json_schema": {"schema": schema},
    "strict": True
}
```

For Ollama, Synalinks automatically switches from `ollama/` to `ollama_chat/` prefix for better performance with chat-style prompts.

---

## Gemini / XAI

Google Gemini and XAI (Grok) models use JSON schema response formats similar to Ollama/Mistral.

---

## vLLM

For self-hosted vLLM deployments, use the `vllm/` or `hosted_vllm/` prefix. Synalinks defaults to `http://localhost:8000` as the API base, configurable via the `HOSTED_VLLM_API_BASE` environment variable.

---

## Adding New Providers

If you encounter an unsupported provider, Synalinks will raise a clear error:

```
LM provider 'provider_name' not supported yet, please ensure that
they support constrained structured output and fill an issue.
```

To add support for a new provider:

1. Verify the provider supports constrained structured output (JSON schema or tool-based)
2. Open an issue or PR on the Synalinks repository
3. Include documentation links showing the provider's structured output API
