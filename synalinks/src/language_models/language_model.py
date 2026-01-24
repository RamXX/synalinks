# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import ast
import copy
import json
import os
import re
import warnings

import litellm
import orjson
import jsonschema

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatRole
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.nlp_utils import shorten_text

litellm.drop_params = True
litellm.disable_aiohttp_transport = True


@synalinks_export(
    [
        "synalinks.LanguageModel",
        "synalinks.language_models.LanguageModel",
    ]
)
class LanguageModel(SynalinksSaveable):
    """A language model API wrapper.

    A language model is a type of AI model designed to generate, and interpret human
    language. It is trained on large amounts of text data to learn patterns and
    structures in language. Language models can perform various tasks such as text
    generation, translation, summarization, and answering questions.

    We support providers that implement *constrained structured output*
    like OpenAI, Azure, Ollama, Mistral, Groq, or Anthropic.

    For the complete list of models, please refer to the providers documentation.

    **Using OpenAI models**

    ```python
    import synalinks
    import os

    os.environ["OPENAI_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="openai/gpt-4o-mini",
    )
    ```

    **Using Groq models**

    ```python
    import synalinks
    import os

    os.environ["GROQ_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="groq/llama3-8b-8192",
    )
    ```

    **Using Anthropic models**

    ```python
    import synalinks
    import os

    os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="anthropic/claude-3-sonnet-20240229",
    )
    ```

    **Using Mistral models**

    ```python
    import synalinks
    import os

    os.environ["MISTRAL_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="mistral/codestral-latest",
    )
    ```

    **Using Ollama models**

    ```python
    import synalinks
    import os

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )
    ```

    **Using Azure OpenAI models**

    ```python
    import synalinks
    import os

    os.environ["AZURE_API_KEY"] = "your-api-key"
    os.environ["AZURE_API_BASE"] = "your-api-key"
    os.environ["AZURE_API_VERSION"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="azure/<your_deployment_name>",
    )
    ```

    **Using Google Gemini models**

    ```python
    import synalinks
    import os

    os.environ["GEMINI_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="gemini/gemini-2.5-pro",
    )
    ```

    **Using XAI models**

    ```python
    import synalinks
    import os

    os.environ["XAI_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="xai/grok-code-fast-1",
    )
    ```

    To cascade models in case there is anything wrong with
    the model provider (hence making your pipelines more robust).
    Use the `fallback` argument like in this example:

    ```python
    import synalinks
    import os

    os.environ["OPENAI_API_KEY"] = "your-api-key"
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="anthropic/claude-3-sonnet-20240229",
        fallback=synalinks.LanguageModel(
            model="openai/gpt-4o-mini",
        )
    )
    ```

    **Note**: Obviously, use an `.env` file and `.gitignore` to avoid
    putting your API keys in the code or a config file that can lead to
    leackage when pushing it into repositories.

    Args:
        model (str): The model to use.
        api_base (str): Optional. The endpoint to use.
        timeout (int): Optional. The timeout value in seconds (Default to 600).
        retry (int): Optional. The number of retry (default to 2).
        fallback (LanguageModel): Optional. The language model to fallback
            if anything is wrong.
        caching (bool): Optional. Enable caching of LM calls (Default to False).
        max_tokens (int): Optional. Max output tokens for each request.
    """

    # Groq defaults tuned for high-volume structured outputs.
    _GROQ_DEFAULT_MAX_TOKENS = {
        "moonshotai/kimi-k2-instruct-0905": 16384,
        "openai/gpt-oss-20b": 65536,
        "openai/gpt-oss-120b": 65536,
    }
    # Z.AI (OpenAI-compatible) defaults.
    _ZAI_DEFAULT_MAX_TOKENS = 16384
    _GROQ_JSON_GUARD = (
        "Return ONLY a valid JSON object that matches the schema. "
        "Do not include markdown, headings, code fences, or extra text. "
        "All schema keys are required; include them even if empty."
    )

    def __init__(
        self,
        model=None,
        api_base=None,
        timeout=600,
        retry=2,
        fallback=None,
        caching=False,
        max_tokens=None,
    ):
        if model is None:
            raise ValueError("You need to set the `model` argument for any LanguageModel")
        model_provider = model.split("/")[0]
        if model_provider == "ollama":
            # Switch from `ollama` to `ollama_chat`
            # because it have better performance due to the chat prompts
            model = model.replace("ollama", "ollama_chat")
        if model_provider == "vllm":
            model = model.replace("vllm", "hosted_vllm")
        self.model = model
        self.fallback = fallback
        if self.model.startswith("ollama") and not api_base:
            self.api_base = "http://localhost:11434"
        else:
            self.api_base = api_base
        if self.model.startswith("hosted_vllm") and not api_base:
            self.api_base = os.environ.get(
                "HOSTED_VLLM_API_BASE", "http://localhost:8000"
            )
        self.timeout = timeout
        self.retry = retry
        self.caching = caching
        if max_tokens is None:
            inferred_max_tokens = self._infer_max_tokens(self.model, self.api_base)
            if inferred_max_tokens is not None:
                max_tokens = inferred_max_tokens
        self.max_tokens = max_tokens
        self.cumulated_cost = 0.0
        self.last_call_cost = 0.0

    @classmethod
    def _default_groq_max_tokens(cls, model_name: str):
        if not model_name.startswith("groq/"):
            return None
        model_key = model_name.split("/", 1)[1]
        return cls._GROQ_DEFAULT_MAX_TOKENS.get(model_key)

    @classmethod
    def _infer_max_tokens(cls, model_name: str, api_base: str | None):
        try:
            info = litellm.get_model_info(model=model_name)
            if isinstance(info, dict):
                max_out = info.get("max_output_tokens") or info.get("max_tokens")
            else:
                max_out = getattr(info, "max_output_tokens", None) or getattr(
                    info, "max_tokens", None
                )
            if max_out:
                return max_out
        except Exception:
            pass
        groq_default = cls._default_groq_max_tokens(model_name)
        if groq_default is not None:
            return groq_default
        if api_base and "z.ai" in api_base:
            return cls._ZAI_DEFAULT_MAX_TOKENS
        return None

    @classmethod
    def _inject_groq_json_guard(cls, messages, schema=None):
        guard = cls._GROQ_JSON_GUARD
        if schema and isinstance(schema, dict):
            props = schema.get("properties", {}) or {}
            if "code" in props or "code_lines" in props:
                guard = (
                    f"{guard} Output only the action keys (reasoning + code/code_lines); "
                    "do not output final result fields as JSON keys. Use single quotes "
                    "inside code fields; avoid double quotes. Never include literal "
                    "backslashes in code fields; build them via BACKSLASH/chr(92) if "
                    "absolutely needed."
                )
            if "reasoning" in props:
                guard = (
                    f"{guard} Always include the reasoning field (use an empty string if needed)."
                )
        if not messages:
            return [{"role": "system", "content": guard}]
        if messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            if guard not in content:
                messages[0]["content"] = f"{guard}\n\n{content}".strip()
        else:
            messages.insert(0, {"role": "system", "content": guard})
        return messages

    @staticmethod
    def _clean_messages_for_groq(messages):
        """Clean message fields for Groq API compatibility.

        Groq's API enforces strict message schema validation and rejects messages
        that contain fields not applicable to their role type. Specifically:
        - system/user messages must NOT contain 'tool_calls' or 'tool_call_id'
        - Only assistant messages may have 'tool_calls'
        - Only tool messages may have 'tool_call_id'

        This method strips invalid fields from each message based on its role.

        Args:
            messages (list): List of message dicts with role, content, etc.

        Returns:
            list: Cleaned messages with only role-appropriate fields.
        """
        cleaned = []
        for msg in messages:
            role = msg.get("role")
            if role in ("system", "user"):
                # System and user messages: only role and content
                clean_msg = {"role": role, "content": msg.get("content", "")}
            elif role == "assistant":
                # Assistant messages: role, content, and optionally tool_calls
                clean_msg = {"role": role, "content": msg.get("content", "")}
                if msg.get("tool_calls"):
                    clean_msg["tool_calls"] = msg["tool_calls"]
            elif role == "tool":
                # Tool messages: role, content, tool_call_id, and name
                clean_msg = {
                    "role": role,
                    "content": msg.get("content", ""),
                    "tool_call_id": msg.get("tool_call_id"),
                    "name": msg.get("name"),
                }
            else:
                # Unknown role: pass through as-is
                clean_msg = msg
            cleaned.append(clean_msg)
        return cleaned

    _VALID_JSON_ESCAPES = set(['"', "\\", "/", "b", "f", "n", "r", "t", "u"])

    @classmethod
    def _strip_json_code_fences(cls, text: str) -> str:
        if "```" not in text:
            return text
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if blocks:
            return blocks[0].strip()
        return text

    @classmethod
    def _escape_invalid_json_backslashes(cls, text: str) -> str:
        result = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch != "\\":
                result.append(ch)
                i += 1
                continue
            if i + 1 >= len(text):
                result.append("\\\\")
                i += 1
                continue
            nxt = text[i + 1]
            if nxt == "u":
                if i + 5 < len(text) and all(
                    c in "0123456789abcdefABCDEF" for c in text[i + 2 : i + 6]
                ):
                    result.append(ch)
                    result.append(nxt)
                    result.extend(text[i + 2 : i + 6])
                    i += 6
                    continue
                result.append("\\\\")
                i += 1
                continue
            if nxt in cls._VALID_JSON_ESCAPES:
                result.append(ch)
                result.append(nxt)
                i += 2
                continue
            result.append("\\\\")
            i += 1
        return "".join(result)

    @classmethod
    def _extract_first_json_object(cls, text: str) -> str | None:
        start = None
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]
        return None

    @classmethod
    def _parse_json_candidate(cls, text: str):
        if not text:
            return None
        candidate = cls._strip_json_code_fences(text.strip())
        attempts = [
            candidate,
            cls._escape_invalid_json_backslashes(candidate),
        ]
        for attempt in attempts:
            try:
                return orjson.loads(attempt)
            except Exception:
                try:
                    return json.loads(attempt, strict=False)
                except Exception:
                    continue
        extracted = cls._extract_first_json_object(candidate)
        if extracted:
            extracted_attempts = [
                extracted,
                cls._escape_invalid_json_backslashes(extracted),
            ]
            for attempt in extracted_attempts:
                try:
                    return orjson.loads(attempt)
                except Exception:
                    try:
                        return json.loads(attempt, strict=False)
                    except Exception:
                        continue
        try:
            return ast.literal_eval(candidate)
        except Exception:
            return None

    @classmethod
    def _drop_unknown_keys(cls, schema: dict, value):
        if not isinstance(schema, dict):
            return value
        schema_type = schema.get("type")
        if schema_type == "object" and isinstance(value, dict):
            props = schema.get("properties", {}) or {}
            cleaned = {k: value[k] for k in value if k in props}
            for key, subschema in props.items():
                if key in cleaned:
                    cleaned[key] = cls._drop_unknown_keys(subschema, cleaned[key])
            return cleaned
        if schema_type == "array" and isinstance(value, list):
            items_schema = schema.get("items")
            if items_schema:
                return [cls._drop_unknown_keys(items_schema, item) for item in value]
        return value

    @classmethod
    def _coerce_to_schema(cls, value, schema: dict):
        if not isinstance(schema, dict):
            return value
        schema_type = schema.get("type")
        if schema_type == "array":
            if value is None:
                return []
            if isinstance(value, list):
                items_schema = schema.get("items")
                if items_schema:
                    return [cls._coerce_to_schema(v, items_schema) for v in value]
                return value
            return [value]
        if schema_type == "string":
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join(map(str, value))
            return str(value)
        if schema_type == "integer":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except Exception:
                    return value
        if schema_type == "number":
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except Exception:
                    return value
        if schema_type == "boolean":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "false"):
                    return lowered == "true"
            return bool(value)
        if schema_type == "object" and isinstance(value, dict):
            props = schema.get("properties", {}) or {}
            coerced = {}
            for key, subschema in props.items():
                if key in value:
                    coerced[key] = cls._coerce_to_schema(value[key], subschema)
            for key in value:
                if key not in coerced:
                    coerced[key] = value[key]
            return coerced
        return value

    @classmethod
    def _finalize_structured_output(cls, schema: dict, value):
        if not isinstance(value, dict):
            return None, jsonschema.ValidationError("Structured output is not an object")
        normalized = cls._drop_unknown_keys(schema, value)
        normalized = cls._coerce_to_schema(normalized, schema)
        try:
            jsonschema.validate(instance=normalized, schema=schema)
        except jsonschema.ValidationError as exc:
            return normalized, exc
        return normalized, None

    @staticmethod
    def _extract_groq_failed_generation(error: Exception) -> str | None:
        """Extract Groq failed_generation payload from an exception message."""
        text = str(error)
        if "failed_generation" not in text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        payload_str = text[start : end + 1]
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            return None
        return payload.get("error", {}).get("failed_generation")

    @staticmethod
    def _build_submit_call(values: dict) -> str:
        """Build a SUBMIT(...) call from a dict of output values."""
        if not values:
            return "SUBMIT()"
        parts = []
        for key in sorted(values.keys()):
            if key in ("reasoning", "code", "code_lines"):
                continue
            parts.append(f"{key}={repr(values[key])}")
        if not parts:
            return "SUBMIT()"
        return f"SUBMIT({', '.join(parts)})"

    @classmethod
    def _recover_groq_failed_generation(cls, schema: dict, error: Exception) -> dict | None:
        """Attempt to recover a valid action from Groq json_validate_failed errors."""
        failed = cls._extract_groq_failed_generation(error)
        if not failed:
            return None
        parsed = cls._parse_json_candidate(failed)
        if parsed is None:
            props = schema.get("properties", {}) or {}
            if "code_lines" in props:
                return {"reasoning": "", "code_lines": ["print('retry')"]}
            if "code" in props:
                return {"reasoning": "", "code": "print('retry')"}
            return None
        if not isinstance(parsed, dict):
            props = schema.get("properties", {}) or {}
            if "code_lines" in props:
                return {"reasoning": "", "code_lines": ["print('retry')"]}
            if "code" in props:
                return {"reasoning": "", "code": "print('retry')"}
            return None

        props = schema.get("properties", {}) or {}
        if "code" not in props and "code_lines" not in props:
            return None

        result = dict(parsed)
        if "code_lines" in props and "code_lines" not in result and "code" not in result:
            submit_code = cls._build_submit_call(result)
            result = {"reasoning": result.get("reasoning", ""), "code_lines": [submit_code]}
        elif "code" in props and "code" not in result and "code_lines" not in result:
            submit_code = cls._build_submit_call(result)
            result = {"reasoning": result.get("reasoning", ""), "code": submit_code}

        if "reasoning" in props and "reasoning" not in result:
            result["reasoning"] = ""

        if "code_lines" in props and "code_lines" in result:
            if isinstance(result["code_lines"], str):
                result["code_lines"] = [result["code_lines"]]
            elif not isinstance(result["code_lines"], list):
                result["code_lines"] = [str(result["code_lines"])]
            else:
                result["code_lines"] = [str(line) for line in result["code_lines"]]

        if "code" in props and "code" in result and not isinstance(result["code"], str):
            result["code"] = str(result["code"])

        # Drop extra keys to satisfy additionalProperties=false.
        result = {key: value for key, value in result.items() if key in props}

        for key in schema.get("required", []) or []:
            if key in result:
                continue
            if key == "reasoning":
                result[key] = ""
            elif key == "code_lines":
                result[key] = []
            elif key == "code":
                result[key] = ""
            else:
                return None

        return result

    async def _repair_structured_output(
        self,
        schema: dict,
        invalid_output: str,
        error_summary: str,
        base_kwargs: dict,
    ) -> dict | None:
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON repair assistant. Return ONLY valid JSON that matches the schema. "
                    "Do not include any extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Schema:\\n"
                    + json.dumps(schema, ensure_ascii=False)
                    + "\\n\\nInvalid Output:\\n"
                    + invalid_output
                    + "\\n\\nError:\\n"
                    + error_summary
                ),
            },
        ]

        if self.model.startswith("groq"):
            repair_messages = self._clean_messages_for_groq(repair_messages)
            repair_messages = self._inject_groq_json_guard(repair_messages, schema)

        kwargs = copy.deepcopy(base_kwargs)
        kwargs.pop("stream", None)
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=repair_messages,
                timeout=self.timeout,
                caching=self.caching,
                **kwargs,
            )
            if hasattr(response, "_hidden_params"):
                if "response_cost" in response._hidden_params:
                    self.last_call_cost = response._hidden_params["response_cost"]
                    if self.last_call_cost is not None:
                        self.cumulated_cost += self.last_call_cost
            response_str = response["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

        parsed = self._parse_json_candidate(response_str)
        if parsed is None:
            return None
        normalized, error = self._finalize_structured_output(schema, parsed)
        if error:
            return None
        return normalized

    @staticmethod
    def _enforce_required_properties(schema):
        """Recursively require all properties on object schemas (Groq)."""
        if isinstance(schema, list):
            for item in schema:
                LanguageModel._enforce_required_properties(item)
            return schema

        if not isinstance(schema, dict):
            return schema

        schema_type = schema.get("type")
        if schema_type == "object":
            properties = schema.get("properties", {}) or {}
            if properties:
                required = set(schema.get("required", []))
                required.update(properties.keys())
                schema["required"] = sorted(required)
            for prop in properties.values():
                LanguageModel._enforce_required_properties(prop)
            for key in ("$defs", "definitions"):
                if key in schema and isinstance(schema[key], dict):
                    for def_schema in schema[key].values():
                        LanguageModel._enforce_required_properties(def_schema)
        elif schema_type == "array":
            if "items" in schema:
                LanguageModel._enforce_required_properties(schema["items"])

        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema and isinstance(schema[key], list):
                LanguageModel._enforce_required_properties(schema[key])

        return schema

    @staticmethod
    def _enforce_no_additional_properties(schema):
        """Recursively set additionalProperties=false on object schemas."""
        if isinstance(schema, list):
            for item in schema:
                LanguageModel._enforce_no_additional_properties(item)
            return schema

        if not isinstance(schema, dict):
            return schema

        schema_type = schema.get("type")
        if schema_type == "object":
            schema["additionalProperties"] = False
            for prop in schema.get("properties", {}).values():
                LanguageModel._enforce_no_additional_properties(prop)
            for key in ("$defs", "definitions"):
                if key in schema and isinstance(schema[key], dict):
                    for def_schema in schema[key].values():
                        LanguageModel._enforce_no_additional_properties(def_schema)
        elif schema_type == "array":
            if "items" in schema:
                LanguageModel._enforce_no_additional_properties(schema["items"])

        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema and isinstance(schema[key], list):
                LanguageModel._enforce_no_additional_properties(schema[key])

        return schema

    async def __call__(self, messages, schema=None, streaming=False, **kwargs):
        """
        Call method to generate a response using the language model.

        Args:
            messages (dict): A formatted dict of chat messages.
            schema (dict): The target JSON schema for structed output (optional).
                If None, output a ChatMessage-like answer.
            streaming (bool): Enable streaming (optional). Default to False.
                Can be enabled only if schema is None.
            **kwargs (keyword arguments): The additional keywords arguments
                forwarded to the LM call.
        Returns:
            (dict): The generated structured response.
        """
        formatted_messages = messages.get_json().get("messages", [])

        # Groq requires strict message schema - clean fields per role type
        if self.model.startswith("groq"):
            formatted_messages = self._clean_messages_for_groq(formatted_messages)
            if schema:
                formatted_messages = self._inject_groq_json_guard(formatted_messages, schema)
        json_instance = {}
        input_kwargs = copy.deepcopy(kwargs)
        schema = copy.deepcopy(schema)

        # Handle reasoning_effort parameter
        reasoning_effort = kwargs.pop("reasoning_effort", "none")
        use_reasoning = reasoning_effort not in ("none", "disable")
        thinking_removed = False

        # Check if the model supports reasoning before using it
        if use_reasoning:
            if not litellm.supports_reasoning(model=self.model):
                use_reasoning = False

        # Check if the model actually exposes reasoning_content in the response.
        # Some models (like OpenAI) support the reasoning_effort parameter but
        # do NOT return reasoning_content in the API response.
        # Only Anthropic models expose reasoning content via thinking_blocks.
        model_exposes_reasoning = self.model.startswith("anthropic")

        # If reasoning_effort is active AND the model exposes reasoning content,
        # remove "thinking" field from schema - the LM will use its internal
        # reasoning (reasoning_content) instead.
        # For models that don't expose reasoning (like OpenAI), keep the "thinking"
        # field so the model generates it as part of structured output.
        if use_reasoning and model_exposes_reasoning and schema:
            if "properties" in schema and "thinking" in schema["properties"]:
                del schema["properties"]["thinking"]
                thinking_removed = True
            if "required" in schema and "thinking" in schema["required"]:
                schema["required"] = [r for r in schema["required"] if r != "thinking"]

        # Pass reasoning_effort to LiteLLM when reasoning is enabled
        if use_reasoning:
            kwargs["reasoning_effort"] = reasoning_effort

        # Encourage deterministic structured output for Groq
        if schema and self.model.startswith("groq") and "temperature" not in kwargs:
            kwargs["temperature"] = 0

        # Apply default max_tokens if configured and not overridden per-call
        if self.max_tokens is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens

        if schema:
            if self.model.startswith("groq"):
                # Use json_schema response format for Groq
                # Groq requires additionalProperties: false on all object schemas
                groq_schema = copy.deepcopy(schema)
                groq_schema = self._enforce_no_additional_properties(groq_schema)
                groq_schema = self._enforce_required_properties(groq_schema)
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_output",
                                "schema": groq_schema,
                                "strict": True,
                            },
                        },
                    }
                )
            elif self.model.startswith("anthropic"):
                # Use response_format for Anthropic - LiteLLM handles this correctly:
                # - For newer models (sonnet-4.5, opus-4.1): uses native output_format
                # - For older models: uses tool call with proper tool_choice handling
                #   (auto when thinking is enabled, forced otherwise)
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema,
                            },
                        },
                    }
                )
            elif self.model.startswith("ollama") or self.model.startswith("mistral"):
                # Use constrained structured output for ollama/mistral
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"schema": schema},
                            "strict": True,
                        },
                    }
                )
            elif self.model.startswith("openai") or self.model.startswith("azure"):
                # Use constrained structured output for openai
                # OpenAI require the field  "additionalProperties"
                # Also OpenAI disallow the field "description" in $ref
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
            elif self.model.startswith("gemini"):
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema,
                            },
                            "strict": True,
                        }
                    }
                )
            elif self.model.startswith("xai"):
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema,
                            },
                            "strict": True,
                        }
                    }
                )
            elif self.model.startswith("hosted_vllm"):
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema,
                            },
                            "strict": True,
                        }
                    }
                )
            else:
                provider = self.model.split("/")[0]
                raise ValueError(
                    f"LM provider '{provider}' not supported yet, please ensure that"
                    " they support constrained structured output and fill an issue."
                )

        if "api_key" not in kwargs:
            if self.api_base and "z.ai" in self.api_base:
                zai_key = os.environ.get("ZAI_API_KEY")
                if zai_key:
                    kwargs["api_key"] = zai_key
        if self.api_base:
            kwargs.update(
                {
                    "api_base": self.api_base,
                }
            )
        if streaming and schema:
            streaming = False
        if streaming:
            kwargs.update({"stream": True})
        base_kwargs = copy.deepcopy(kwargs)
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
                if hasattr(response, "_hidden_params"):
                    if "response_cost" in response._hidden_params:
                        self.last_call_cost = response._hidden_params["response_cost"]
                        if self.last_call_cost is not None:
                            self.cumulated_cost += self.last_call_cost
                if streaming:
                    return StreamingIterator(response)
                # All providers use response_format which returns content in message["content"]
                response_str = response["choices"][0]["message"]["content"].strip()
                if schema:
                    parsed = self._parse_json_candidate(response_str)
                    if parsed is None:
                        repaired = await self._repair_structured_output(
                            schema=schema,
                            invalid_output=response_str,
                            error_summary="Failed to parse JSON response",
                            base_kwargs=base_kwargs,
                        )
                        if repaired is None:
                            raise ValueError("Failed to parse structured JSON response")
                        json_instance = repaired
                    else:
                        normalized, error = self._finalize_structured_output(schema, parsed)
                        if error:
                            repaired = await self._repair_structured_output(
                                schema=schema,
                                invalid_output=response_str,
                                error_summary=str(error),
                                base_kwargs=base_kwargs,
                            )
                            if repaired is None:
                                raise error
                            json_instance = repaired
                        else:
                            json_instance = normalized
                    # If reasoning_effort is active and thinking was removed from schema,
                    # populate the "thinking" field from reasoning_content
                    if use_reasoning and thinking_removed:
                        message = response["choices"][0]["message"]
                        reasoning_content = getattr(message, "reasoning_content", None)
                        # Always set thinking field if it was removed, default to empty
                        json_instance = {"thinking": reasoning_content or "", **json_instance}
                else:
                    json_instance = {
                        "role": ChatRole.ASSISTANT,
                        "content": response_str,
                        "tool_call_id": None,
                        "tool_calls": [],
                    }
                return json_instance
            except Exception as e:
                if schema and self.model.startswith("groq"):
                    recovered = self._recover_groq_failed_generation(schema, e)
                    if recovered is not None:
                        normalized, error = self._finalize_structured_output(schema, recovered)
                        if error is None:
                            return normalized
                        repaired = await self._repair_structured_output(
                            schema=schema,
                            invalid_output=json.dumps(recovered, ensure_ascii=False),
                            error_summary=str(error),
                            base_kwargs=base_kwargs,
                        )
                        if repaired is not None:
                            return repaired
                warnings.warn(
                    f"Error occured while trying to call {self}: "
                    + str(e)
                    + f"\nReceived response={shorten_text(response_str)}"
                )
            await asyncio.sleep(1)
        if self.fallback:
            return await self.fallback(
                messages,
                schema=schema,
                streaming=streaming,
                **input_kwargs,
            )
        else:
            return None

    def _obj_type(self):
        return "LanguageModel"

    def get_config(self):
        config = {
            "model": self.model,
            "api_base": self.api_base,
            "timeout": self.timeout,
            "retry": self.retry,
            "caching": self.caching,
            "max_tokens": self.max_tokens,
        }
        if self.fallback:
            fallback_config = {
                "fallback": serialization_lib.serialize_synalinks_object(
                    self.fallback,
                )
            }
            return {**fallback_config, **config}
        else:
            return config

    @classmethod
    def from_config(cls, config):
        if "fallback" in config:
            fallback = serialization_lib.deserialize_synalinks_object(
                config.pop("fallback")
            )
            return cls(fallback=fallback, **config)
        else:
            return cls(**config)

    def __repr__(self):
        api_base = f" api_base={self.api_base}" if self.api_base else ""
        return f"<LanguageModel model={self.model}{api_base}>"


class StreamingIterator:
    def __init__(self, iterator):
        self._iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        content = self._iterator.__next__()["choices"][0]["delta"]["content"]
        if content:
            return {"role": ChatRole.ASSISTANT, "content": content}
        else:
            raise StopIteration
