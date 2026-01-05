# RLM Integration: Provider Support and Findings

This document explains the work done to support multiple LLM providers for the Recursive Language Model (RLM) and ChainOfThought modules in Synalinks.

## Overview

The RLM (Recursive Language Model) approach allows LLMs to execute Python code in a REPL environment, enabling them to verify calculations and perform complex reasoning through code execution. This is implemented via `RecursiveGenerator` and `RecursiveChainOfThought` modules.

## Provider Support Added

### 1. Groq OpenAI-Compatible Models (`groq/openai/*`)

Models like `groq/openai/gpt-oss-120b` are OpenAI-compatible models hosted on Groq's infrastructure.

**Issues Found:**
- Groq API rejects messages containing `tool_calls` or `tool_call_id` properties (even when empty)
- These models don't support Groq's native tool_choice mechanism
- They support `json_schema` response format instead

**Solution:**
```python
# Message cleaning for Groq
def _clean_messages_for_groq(messages: list) -> list:
    return [{"role": m.get("role"), "content": m.get("content", "")} for m in messages]

# Use json_schema for groq/openai/* models
if self.model.startswith("groq/openai"):
    kwargs.update({
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "schema": schema},
        },
    })
```

### 2. OpenRouter (`openrouter/*`)

OpenRouter provides access to multiple models through a unified API.

**Issues Found:**
- Requires message cleaning (same as Groq)
- Uses `json_schema` response format

**Solution:**
```python
# Added to message cleaning
if self.model.startswith("openrouter"):
    formatted_messages = _clean_messages_for_groq(formatted_messages)

# Schema handling
elif self.model.startswith("openrouter"):
    kwargs.update({
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "schema": schema},
        }
    })
```

### 3. z.ai (`zai/*`)

z.ai provides Anthropic-compatible API access to models like `glm-4.7`.

**Issues Found:**
- Uses Anthropic-compatible API at `https://api.z.ai/api/anthropic`
- Does NOT support tool_choice through LiteLLM (unlike native Anthropic)
- Supports `json_schema` response format
- Requires `ZAI_API_KEY` environment variable

**Solution:**
```python
# Model transformation
if model_provider == "zai":
    model = model.replace("zai/", "anthropic/")

# API base setting
if model_provider == "zai" and not api_base:
    self.api_base = "https://api.z.ai/api/anthropic"

# Schema handling (json_schema, not tool_choice)
elif self._original_provider == "zai":
    kwargs.update({
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "schema": schema},
        },
    })
```

**Usage:**
```python
import os
os.environ["ANTHROPIC_API_KEY"] = os.environ["ZAI_API_KEY"]

lm = synalinks.LanguageModel(model="zai/glm-4.7")
```

## Bug Fixes

### Cost Tracking None Handling

Fixed `TypeError: unsupported operand type(s) for +=: 'float' and 'NoneType'` when response cost is None:

```python
if hasattr(response, "_hidden_params"):
    if "response_cost" in response._hidden_params:
        cost = response._hidden_params["response_cost"]
        if cost is not None:  # Added None check
            self.last_call_cost = cost
            self.cumulated_cost += self.last_call_cost
```

## Test Results

### ChainOfThought Accuracy

| Model | Accuracy | Notes |
|-------|----------|-------|
| zai/glm-4.7 | 100% (5/5) | Excellent arithmetic |
| groq/openai/gpt-oss-120b | 100% (5/5) | Good arithmetic |
| openrouter/meta-llama/llama-3.1-8b-instruct | ~60% | Makes arithmetic mistakes |

### RecursiveChainOfThought Findings

The RLM approach requires models to follow a specific protocol:
1. Write code in ` ```repl ` blocks
2. Execute code to verify calculations
3. Call `FINAL_VAR(result)` to return structured output

**Why RLM Doesn't Currently Beat CoT:**

1. **Protocol Compliance**: Most LLMs don't follow the ` ```repl ` + `FINAL_VAR()` protocol

2. **Smart Models Answer Directly**: z.ai (glm-4.7) outputs JSON answers immediately without using code execution - it's too smart and bypasses the REPL

3. **Model-Specific Issues**:
   - groq/openai/gpt-oss-120b hallucinates tool calls (`repo_browser.open_file`)
   - Llama 3.1 8B uses ` ```python ` blocks instead of ` ```repl `

**For RLM to Be Effective:**
- Models need to be specifically trained to use the REPL protocol
- Or use models that make arithmetic mistakes but can follow the protocol for verification

## Usage Examples

### Running Arithmetic Test

```bash
# Default (z.ai)
uv run --env-file .env -- python examples/rlm_comparison/arithmetic_accuracy_test.py

# Specify model via environment
SYNALINKS_MODEL=groq/openai/gpt-oss-120b uv run --env-file .env -- python examples/rlm_comparison/arithmetic_accuracy_test.py

SYNALINKS_MODEL=openrouter/meta-llama/llama-3.1-8b-instruct uv run --env-file .env -- python examples/rlm_comparison/arithmetic_accuracy_test.py
```

### Environment Variables Required

```bash
# For z.ai models
ZAI_API_KEY=your_key

# For Groq models
GROQ_API_KEY=your_key

# For OpenRouter models
OPENROUTER_API_KEY=your_key
```

## Files Modified

1. **`synalinks/src/language_models/language_model.py`**
   - Added `_clean_messages_for_groq()` function
   - Added z.ai provider support (model transformation, API base)
   - Added groq/openai/* json_schema handling
   - Added OpenRouter support
   - Fixed cost tracking None handling
   - Updated response parsing for providers using json_schema

2. **`examples/rlm_comparison/arithmetic_accuracy_test.py`**
   - Fixed `.json` attribute access to `.get_json()` method
   - Added model selection via `SYNALINKS_MODEL` environment variable
   - Added provider-specific API key validation
   - Fixed lint issues (unused imports, f-string placeholders)

## Architecture Summary

```
LanguageModel.__call__()
    |
    +-- Message Cleaning (Groq/OpenRouter/z.ai)
    |       Remove tool_calls, tool_call_id from messages
    |
    +-- Schema Handling by Provider:
    |       groq/openai/*  -> json_schema response_format
    |       groq/*         -> tool_choice (native Groq)
    |       zai/*          -> json_schema response_format
    |       anthropic/*    -> tool_choice (Anthropic)
    |       openrouter/*   -> json_schema response_format
    |       openai/*       -> json_schema response_format
    |       ...
    |
    +-- Response Parsing:
            tool_choice providers -> extract from tool_calls
            json_schema providers -> extract from content
```

## Future Work

1. **RLM Protocol Training**: Fine-tune models to follow the ` ```repl ` + `FINAL_VAR()` protocol
2. **Hybrid Approach**: Detect when models output JSON directly and accept it as valid
3. **Provider Auto-Detection**: Automatically detect provider capabilities and choose optimal structured output method
