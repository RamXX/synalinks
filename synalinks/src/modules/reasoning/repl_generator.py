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


# Action instructions template matching DSPy's behavioral guidance
ACTION_INSTRUCTIONS_TEMPLATE = """Return ONLY a JSON object with keys `reasoning` and `code` (both strings). No markdown, no labels, no extra keys.

You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: see `variables_info` in the input context (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, etc.
{tool_docs}
IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

## Rules

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.
7. JSON SAFETY - Your response must be valid JSON. Avoid unescaped double quotes inside the `code` string. Prefer single quotes in code, avoid triple-quoted strings, and if you must use a double quote inside code, escape it with a backslash.
8. BACKSLASH SAFETY - Avoid backslashes in code (e.g., regex patterns or escape sequences). If you must include a backslash, build it via `chr(92)` or string concatenation to prevent invalid JSON escapes.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output.

## Output Fields Required
{output_fields_list}"""


def get_repl_instructions(
    output_fields: List[str],
    tool_descriptions: str = "",
    max_llm_calls: int = 50,
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

    # Format output fields as bullet list
    output_fields_formatted = "\n".join(f"- {n}: <description>" for n in output_fields)
    output_fields_list = ", ".join(output_fields)

    # Format tool docs section
    tool_docs = f"\n{tool_descriptions}\n" if tool_descriptions else ""

    return ACTION_INSTRUCTIONS_TEMPLATE.format(
        inputs=inputs_placeholder,
        output_fields=output_fields_formatted,
        final_output_names=submit_args,
        tool_docs=tool_docs,
        max_llm_calls=max_llm_calls,
        output_fields_list=output_fields_list,
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

        action_schema = REPLAction.get_schema()
        if self._supports_direct_output(language_model):
            action_schema = self._build_action_schema(action_schema, output_schema)

        if not instructions:
            instructions = get_repl_instructions(
                self.output_fields, tool_descriptions, max_llm_calls
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
