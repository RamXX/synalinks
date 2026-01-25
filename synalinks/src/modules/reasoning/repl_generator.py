# License Apache 2.0: (c) 2025 Synalinks Team

"""REPLGenerator module for REPL code generation.

This module is based on the Reasoning Language Models (RLM) implementation
from DSPy (https://github.com/stanfordnlp/dspy). The action instructions
template and behavioral rules were adapted from DSPy's RLM implementation.

Reference:
    DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
    https://github.com/stanfordnlp/dspy
"""

import copy
import re
from typing import List, Optional

from pydantic import Field

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.pydantic.core import DataModel
from synalinks.src.modules.core.generator import Generator


# Pattern to match markdown code fences: ```python\n...\n``` or ```\n...\n```
_CODE_FENCE_PATTERN = re.compile(r"^```(?:python|py)?\s*\n(.*)\n```\s*$", re.DOTALL)


def strip_code_fences(code: str) -> str:
    """Strip markdown code fences from LLM-generated code.

    LLMs often wrap code in ```python ... ``` blocks. This function
    removes the fences to get the raw code.

    Args:
        code: Code string that may contain markdown fences.

    Returns:
        Code with fences stripped, or original code if no fences found.

    Example:

    ```python
    code = '''```python
    print("hello")
    ```'''
    stripped = strip_code_fences(code)
    # stripped == 'print("hello")'
    ```
    """
    code = code.strip()
    match = _CODE_FENCE_PATTERN.match(code)
    if match:
        return match.group(1)
    return code


@synalinks_export("synalinks.REPLAction")
class REPLAction(DataModel):
    """LLM output for each REPL iteration.

    The LLM generates reasoning about what to do next and
    Python code to execute.

    Attributes:
        reasoning: Step-by-step reasoning about the next action.
        code: Python code to execute in the REPL.
    """

    reasoning: str = Field(
        description="Your step-by-step reasoning about what to do next"
    )
    code: str = Field(
        description="Python code to execute. Use SUBMIT(field=value, ...) when done."
    )


@synalinks_export("synalinks.REPLActionLines")
class REPLActionLines(DataModel):
    """LLM output for each REPL iteration using code lines.

    This variant avoids multiline JSON strings by emitting code as
    a list of single-line statements.
    """

    reasoning: str = Field(
        description="Your step-by-step reasoning about what to do next"
    )
    code_lines: List[str] = Field(
        description=(
            "Python code lines to execute (one line per item). "
            "Use SUBMIT(field=value, ...) when done."
        )
    )


# Action instructions template matching DSPy's behavioral guidance
ACTION_INSTRUCTIONS_TEMPLATE = """Return ONLY a JSON object with keys `reasoning` and `code` (both strings). `reasoning` is REQUIRED (use an empty string if needed). No markdown, no labels, no extra keys. No other keys are allowed; any extra keys will be rejected. Do NOT return final output fields directly; always use SUBMIT(...) inside `code`.

Example JSON:
{{"reasoning": "", "code": "print('ready'); SUBMIT(answer='ok')"}}

You are tasked with producing the following FINAL outputs given the inputs {inputs} (ONLY via SUBMIT, never as JSON keys):
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: see `variables_info` in the input context (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries (preloaded, no imports required): re, json, collections, math
{tool_docs}
IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

## Rules

{rules_section}

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output.

## Output Fields Required (SUBMIT only; never return as JSON keys)
{required_fields_list}"""


ACTION_INSTRUCTIONS_TEMPLATE_LINES = """Return ONLY a JSON object with keys `reasoning` (string) and `code_lines` (array of strings). `reasoning` is REQUIRED (use an empty string if needed). No markdown, no labels, no extra keys. No other keys are allowed; any extra keys will be rejected. Do NOT return final output fields directly; always use SUBMIT(...) inside `code_lines`.

Example JSON:
{{"reasoning": "", "code_lines": ["print('ready')", "SUBMIT(answer='ok')"]}}

You are tasked with producing the following FINAL outputs given the inputs {inputs} (ONLY via SUBMIT, never as JSON keys):
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: see `variables_info` in the input context (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries (preloaded, no imports required): re, json, collections, math
{tool_docs}
IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

## Rules

{rules_section}

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output.

## Output Fields Required (SUBMIT only; never return as JSON keys)
{required_fields_list}"""


_BASE_RULES = [
    "EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.",
    "ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.",
    "ENVIRONMENT LIMITS - Do NOT use import statements. Only preloaded modules are available: re, json, collections, math (use directly). File I/O (open/pathlib/os/glob) is disallowed.",
    "NO MULTI-LINE STRINGS - Do not use triple-quoted or multi-line string literals.",
    "ASCII ONLY - Use ASCII characters only. No smart quotes or em dashes; use '-' for dashes.",
    "SINGLE-LINE CODE - Keep code to a single line; use semicolons to separate statements.",
    "VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.",
    "USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.",
    "MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.",
    "SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.",
]

_STRICT_JSON_RULES = [
    "JSON SAFETY - Your response must be valid JSON. Use single quotes for Python strings (including SUBMIT). Do NOT use double quotes inside code. If you truly need a double quote character, build it with chr(34) and string concatenation.",
    "BACKSLASH SAFETY - Never include literal backslashes in code (regex/escapes will break JSON). If absolutely needed, build them via BACKSLASH (preloaded) or chr(92) and string concatenation. Avoid regex patterns that require backslashes.",
    "NEWLINE SAFETY - Do not include literal newline characters inside JSON strings. If you need multiple statements, separate them with semicolons.",
    "SINGLE-LINE ONLY - In strict JSON mode, keep each code line to a single line; use semicolons to separate statements.",
]

_TAIL_RULES = [
    "SUBMIT WITH VARIABLES - Build outputs in code and pass variables to SUBMIT (e.g., SUBMIT(answer=answer)). Avoid long inline string literals.",
    "NO RAW FILE LITERALS - Never paste file contents into Python string literals. Always slice/print from the provided variables (e.g., print(files[0][:500])).",
    "KEEP CODE SIMPLE - Avoid large nested dict literals or multi-line literals. Prefer simple lists/strings and call SUBMIT with variables.",
]


def _build_rules_section(strict_json: bool) -> str:
    rules = list(_BASE_RULES)
    if strict_json:
        rules.extend(_STRICT_JSON_RULES)
    rules.extend(_TAIL_RULES)
    return "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))


def _schema_type_to_str(schema: dict) -> str:
    if not isinstance(schema, dict):
        return "any"

    if "$ref" in schema:
        return "object"

    schema_type = schema.get("type")
    if schema_type:
        if isinstance(schema_type, list):
            types = schema_type
        else:
            types = [schema_type]
        if "array" in types:
            item_schema = schema.get("items", {})
            item_type = _schema_type_to_str(item_schema)
            return f"array<{item_type}>"
        if len(types) == 1:
            return str(types[0])
        return " | ".join(str(t) for t in types)

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and isinstance(schema[key], list):
            return " | ".join(_schema_type_to_str(item) for item in schema[key])

    return "any"


def _format_output_fields(output_schema: Optional[dict], output_fields: List[str]) -> tuple[str, list[str]]:
    required = set()
    properties = {}
    if isinstance(output_schema, dict):
        required = set(output_schema.get("required") or [])
        properties = output_schema.get("properties", {}) or {}
    if not required:
        required = set(output_fields)

    lines = []
    for name in output_fields:
        prop_schema = properties.get(name, {}) if isinstance(properties, dict) else {}
        type_str = _schema_type_to_str(prop_schema)
        description = prop_schema.get("description", "<description>")
        req_label = "required" if name in required else "optional"
        lines.append(f"- {name} ({req_label}, type={type_str}): {description}")

    return "\n".join(lines), sorted(required)


def get_repl_instructions(
    output_fields: List[str],
    tool_descriptions: str = "",
    max_llm_calls: int = 50,
    use_code_lines: bool = False,
    strict_json: bool = False,
    output_schema: Optional[dict] = None,
) -> str:
    """Generate default REPL instructions with behavioral guidance.

    Args:
        output_fields: List of output field names for SUBMIT.
        tool_descriptions: Formatted string of available tools.
        max_llm_calls: Maximum sub-LLM calls allowed.

    Returns:
        Instruction string for the REPL generator.
    """
    submit_args = ", ".join(f"{f}=value" for f in output_fields)
    inputs_placeholder = "`variables_info`"  # REPL metadata lives in variables_info

    # Format output fields as bullet list (with type hints)
    output_fields_formatted, required_fields = _format_output_fields(
        output_schema, output_fields
    )
    required_fields_list = ", ".join(required_fields)

    # Format tool docs section
    tool_docs = f"\n{tool_descriptions}\n" if tool_descriptions else ""
    rules_section = _build_rules_section(strict_json)

    template = (
        ACTION_INSTRUCTIONS_TEMPLATE_LINES
        if use_code_lines
        else ACTION_INSTRUCTIONS_TEMPLATE
    )
    return template.format(
        inputs=inputs_placeholder,
        output_fields=output_fields_formatted,
        final_output_names=submit_args,
        tool_docs=tool_docs,
        max_llm_calls=max_llm_calls,
        required_fields_list=required_fields_list,
        rules_section=rules_section,
    ).strip()


@synalinks_export(["synalinks.modules.REPLGenerator", "synalinks.REPLGenerator"])
class REPLGenerator(Generator):
    """Generator specialized for REPL code generation.

    Extends Generator with REPL-specific prompt template and
    trainable instructions for iterative code generation.

    The generator produces REPLAction outputs containing reasoning
    and code to execute in each REPL iteration.

    Example:

    ```python
    output_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        }
    }

    generator = REPLGenerator(
        output_schema=output_schema,
        language_model=lm,
    )

    action = await generator(context)
    print(action.get("reasoning"))
    print(action.get("code"))
    ```

    Args:
        output_schema: JSON schema for the final output (used to build SUBMIT instructions).
        language_model: Language model for code generation.
        instructions: Custom instructions (default: auto-generated).
        seed_instructions: Seed instructions for optimization.
        examples: List of example input/output pairs.
        tool_descriptions: Formatted string describing available tools.
        max_llm_calls: Maximum sub-LLM calls allowed (for instructions).
        max_tokens: Optional max output tokens for each LM call.
        name: Module name.
        **kwargs: Additional arguments passed to Generator.
    """

    def __init__(
        self,
        output_schema: dict,
        language_model=None,
        instructions: Optional[str] = None,
        seed_instructions: Optional[List[str]] = None,
        examples: Optional[list] = None,
        tool_descriptions: str = "",
        max_llm_calls: int = 50,
        max_tokens: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.output_schema = output_schema
        self.output_fields = list(output_schema.get("properties", {}).keys())
        self._max_llm_calls = max_llm_calls
        self._strict_json = self._uses_strict_json(language_model)
        self._use_code_lines = self._uses_code_lines(language_model)

        action_schema = REPLAction.get_schema()
        if self._use_code_lines:
            action_schema = REPLActionLines.get_schema()
        if self._supports_direct_output(language_model):
            action_schema = self._build_action_schema(action_schema, output_schema)

        if not instructions:
            instructions = get_repl_instructions(
                self.output_fields,
                tool_descriptions,
                max_llm_calls,
                use_code_lines=self._use_code_lines,
                strict_json=self._strict_json,
                output_schema=self.output_schema,
            )

        super().__init__(
            schema=action_schema,
            language_model=language_model,
            instructions=instructions,
            seed_instructions=seed_instructions,
            examples=examples,
            name=name or "repl_generator",
            use_inputs_schema=True,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config["output_schema"] = self.output_schema
        config["max_llm_calls"] = self._max_llm_calls
        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration dict."""
        from synalinks.src.saving import serialization_lib

        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            language_model=language_model,
            **config,
        )

    @staticmethod
    def _supports_direct_output(language_model) -> bool:
        """Return True when direct output schema is needed for strict providers."""
        model = getattr(language_model, "model", "")
        if not isinstance(model, str):
            return False
        # Groq JSON schema mode is strict; keep schema minimal (REPLAction only)
        # to reduce json_validate_failed errors on complex outputs.
        return False

    @staticmethod
    def _uses_strict_json(language_model) -> bool:
        """Return True when strict JSON guidance should be enabled."""
        return bool(getattr(language_model, "strict_json", False))

    @staticmethod
    def _uses_code_lines(language_model) -> bool:
        """Return True when code_lines output should be used for reliability."""
        return bool(getattr(language_model, "strict_json", False))

    @staticmethod
    def _build_action_schema(action_schema: dict, output_schema: dict) -> dict:
        """Merge REPLAction schema with output schema for strict providers."""
        merged = {
            "type": "object",
            "title": "REPLActionOrOutput",
            "description": "Either REPL action (reasoning+code) or direct output.",
            "additionalProperties": False,
            "properties": {},
            "minProperties": 1,
        }
        merged["properties"].update(copy.deepcopy(action_schema.get("properties", {})))
        merged["properties"].update(copy.deepcopy(output_schema.get("properties", {})))
        # Groq requires required to include every property at top level.
        merged["required"] = sorted(list(merged["properties"].keys()))
        return merged
