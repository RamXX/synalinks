# License Apache 2.0: (c) 2025 Synalinks Team

"""RLM - Recursive Language Model for Synalinks.

This module is based on the Reasoning Language Models (RLM) implementation
from DSPy (https://github.com/stanfordnlp/dspy). The techniques and patterns
used here were reimplemented to integrate with Synalinks' architecture while
preserving the core RLM concepts from the original DSPy implementation.

The class is named RLM to match DSPy's naming convention and the
"Recursive Language Models" paper.

Reference:
    DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
    https://github.com/stanfordnlp/dspy
"""

import asyncio
import io
import keyword
import tokenize
from typing import Any, Callable, Dict, List, Optional, Type

import jsonschema
from pydantic import ValidationError

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.interpreters.base import CodeInterpreter
from synalinks.src.interpreters.native import NativePythonInterpreter
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.modules.reasoning.repl_generator import REPLGenerator
from synalinks.src.modules.reasoning.repl_generator import strip_code_fences
from synalinks.src.modules.reasoning.repl_types import REPLHistory
from synalinks.src.modules.reasoning.repl_types import REPLVariable
from synalinks.src.saving import serialization_lib


@synalinks_export(["synalinks.modules.RLM", "synalinks.RLM", "synalinks.modules.REPLModule", "synalinks.REPLModule"])
class RLM(Module):
    """Recursive Language Model module for Synalinks.

    Based on DSPy's RLM implementation. LLMs write Python code to
    programmatically explore large contexts using iterative code
    execution with sub-LLM queries.

    The RLM enables LLMs to:
    - Write and execute Python code iteratively
    - Query sub-LLMs for semantic analysis of data portions
    - Use custom tools for domain-specific operations
    - Build up state across iterations
    - Submit final structured output

    Pass-by-Reference for Large Data:
        Input variables are passed by reference into the REPL namespace,
        NOT serialized into the prompt. This enables processing of large
        data (e.g., entire books, large documents) without prompt bloat.

        Only a small preview (default 500 chars) appears in the prompt
        to help the LLM understand the data structure. The full data
        is accessible as a Python variable in the REPL environment.

        For example, a 500KB document passed as `document` will:
        - Show a 500-char preview in the prompt context
        - Be fully accessible as `document` variable in code
        - NOT consume 500KB of prompt tokens

    Example:

    ```python
    import synalinks

    class DocumentQuery(synalinks.DataModel):
        document: str = synalinks.Field(description="Document to analyze")
        question: str = synalinks.Field(description="Question to answer")

    class Analysis(synalinks.DataModel):
        answer: str = synalinks.Field(description="Answer to the question")
        evidence: list[str] = synalinks.Field(description="Supporting evidence")

    @synalinks.saving.register_synalinks_serializable()
    async def search_db(query: str) -> dict:
        '''Search the database for relevant information.

        Args:
            query (str): Search query string.
        '''
        return {"results": db.search(query)}

    # Create RLM module
    rlm = synalinks.RLM(
        data_model=Analysis,
        language_model=lm,
        tools=[synalinks.Tool(search_db)],
        max_iterations=20,
    )

    # Use in program
    inputs = synalinks.Input(data_model=DocumentQuery)
    outputs = await rlm(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs)

    # Execute
    result = await program(DocumentQuery(
        document="<very long document>",
        question="What are the key findings?"
    ))
    ```

    Args:
        schema: Output JSON schema (alternative to data_model).
        data_model: Output DataModel class.
        language_model: LLM for code generation.
        sub_language_model: Optional cheaper LLM for llm_query calls.
        interpreter: CodeInterpreter instance (default: NativePythonInterpreter).
        tools: List of Tool modules for custom functionality.
        max_iterations: Maximum REPL iterations (default: 20).
        max_llm_calls: Maximum sub-LLM calls (default: 50).
        max_output_chars: Maximum output characters (default: 100,000).
        max_tokens: Optional max output tokens for each LM call.
        instructions: Custom instructions for code generation.
        seed_instructions: Seed instructions for optimization.
        return_history: Whether to include execution history in output.
        name: Module name.
        description: Module description.
        trainable: Whether the module is trainable.
    """

    def __init__(
        self,
        schema: Optional[dict] = None,
        data_model: Optional[Type] = None,
        language_model=None,
        sub_language_model=None,
        interpreter: Optional[CodeInterpreter] = None,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 100_000,
        max_tokens: Optional[int] = None,
        instructions: Optional[str] = None,
        seed_instructions: Optional[List[str]] = None,
        return_history: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
        trainable: bool = True,
    ):
        super().__init__(
            name=name or "repl_module",
            description=description or "REPL-based reasoning module",
            trainable=trainable,
        )

        # Resolve schema
        if not schema and data_model:
            schema = data_model.get_schema()
        if not schema:
            raise ValueError("Must provide schema or data_model")

        self.output_schema = schema
        self.output_fields = list(schema.get("properties", {}).keys())
        required_fields = schema.get("required")
        self.required_fields = (
            list(required_fields) if isinstance(required_fields, list) else self.output_fields
        )
        self.data_model = data_model

        # Language models
        if not language_model:
            raise ValueError("Must provide language_model")
        self.language_model = language_model
        self.sub_language_model = sub_language_model or language_model

        # Interpreter
        self.interpreter = interpreter or NativePythonInterpreter(
            max_output_chars=max_output_chars
        )

        # Build tool registry
        self.tools: Dict[str, Tool] = {}
        if tools:
            self._validate_tool_names(tools)
            for tool in tools:
                self.tools[tool.name] = tool

        # Configuration
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.max_tokens = max_tokens
        self.return_history = return_history
        self._instructions = instructions
        self._seed_instructions = seed_instructions

        # Build tool descriptions for generator
        tool_descriptions = self._format_tool_descriptions()

        # Trainable code generator
        self.generator = REPLGenerator(
            output_schema=schema,
            language_model=language_model,
            instructions=instructions,
            seed_instructions=seed_instructions,
            tool_descriptions=tool_descriptions,
            max_llm_calls=max_llm_calls,
            max_tokens=max_tokens,
            name=f"generator_{self.name}",
        )

        # Fallback extractor for when max_iterations reached
        self.extractor = Generator(
            schema=schema,
            language_model=language_model,
            instructions=(
                "Extract the final answer from the execution history above. "
                f"Required output fields: {', '.join(self.required_fields)}"
            ),
            max_tokens=max_tokens,
            name=f"extractor_{self.name}",
        )

    _RESERVED_TOOL_NAMES = {"llm_query", "llm_query_batched", "SUBMIT", "print"}

    def _validate_tool_names(self, tools: List[Tool]) -> None:
        """Validate tool names for REPL safety and consistency."""
        seen = set()
        for tool in tools:
            name = tool.name
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(
                    f"Invalid tool name '{name}': must be a valid Python identifier"
                )
            if name in self._RESERVED_TOOL_NAMES:
                raise ValueError(
                    f"Tool name '{name}' conflicts with built-in sandbox function"
                )
            if name in seen:
                raise ValueError(f"Duplicate tool name '{name}'")
            seen.add(name)

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for inclusion in instructions."""
        if not self.tools:
            return ""

        lines = ["## Custom Tools"]
        for name, tool in self.tools.items():
            # Get tool schema for parameter info
            tool_schema = tool.get_input_schema()
            params = tool_schema.get("properties", {})
            param_strs = []
            for param_name, param_info in params.items():
                param_type = param_info.get("type", "any")
                param_strs.append(f"{param_name}: {param_type}")
            params_str = ", ".join(param_strs)
            lines.append(f"- `{name}({params_str})`: {tool.description}")

        return "\n".join(lines)

    def _build_variables(self, inputs) -> List[REPLVariable]:
        """Build REPLVariable list from inputs with field metadata.

        Args:
            inputs: Input data model.

        Returns:
            List of REPLVariable instances with rich metadata.
        """
        if hasattr(inputs, "get_json"):
            data = inputs.get_json()
        else:
            data = dict(inputs)

        variables = []
        for name, value in data.items():
            # Try to get field info from input schema
            field_info = None
            if hasattr(inputs, "model_fields"):
                field_info = inputs.model_fields.get(name)

            variables.append(
                REPLVariable.from_value(name, value, field_info=field_info)
            )
        return variables

    def _format_variables_info(self, variables: List[REPLVariable]) -> str:
        """Format variables for prompt inclusion.

        Args:
            variables: List of REPLVariable instances.

        Returns:
            Formatted string with variable metadata.
        """
        return "\n\n".join(var.format() for var in variables)

    @staticmethod
    def _sanitize_code(code: str) -> str:
        """Best-effort sanitizer for common LLM formatting mistakes."""
        if not code:
            return code
        code = strip_code_fences(code)
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped in ("```", "```python", "```py"):
                continue
            lines.append(line)
        code = "\n".join(lines).strip()
        if (
            (code.startswith("'''") and code.endswith("'''"))
            or (code.startswith('"""') and code.endswith('"""'))
        ):
            code = code[3:-3].strip()
        if len(code) >= 2 and code[0] == code[-1] and code[0] in ("'", '"'):
            inner = code[1:-1]
            if "\\n" in inner or "\\t" in inner:
                try:
                    code = inner.encode("utf-8").decode("unicode_escape")
                except Exception:
                    pass
        return code

    @staticmethod
    def _can_execute_line_by_line(code: str) -> bool:
        """Return True if code has no obvious multi-line blocks."""
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.endswith("\\"):
                return False
            if stripped.endswith(":"):
                return False
            if line.startswith((" ", "\t")):
                return False

        stack = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        except tokenize.TokenError:
            return False

        pairs = {")": "(", "]": "[", "}": "{"}
        for token in tokens:
            if token.type == tokenize.OP:
                if token.string in "([{":
                    stack.append(token.string)
                elif token.string in ")]}":
                    if not stack or stack[-1] != pairs[token.string]:
                        return False
                    stack.pop()
            elif token.type == tokenize.NL and stack:
                return False

        if stack:
            return False

        return True

    async def _execute_line_by_line(
        self,
        code: str,
        input_vars: dict,
        tool_callables: dict,
    ) -> dict:
        """Execute code line by line to recover from syntax errors."""
        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []
        submitted = None
        last_error = None

        for line in code.splitlines():
            if not line.strip():
                continue
            result = await self.interpreter.execute(
                code=line,
                variables=input_vars,
                tools=tool_callables,
            )
            stdout_chunks.append(result.get("stdout", ""))
            if result.get("stderr"):
                stderr_chunks.append(result.get("stderr", ""))
            if result.get("submitted"):
                submitted = result["submitted"]
                break
            if not result.get("success"):
                last_error = result.get("error") or ""
                if last_error:
                    stderr_chunks.append(last_error)

        if submitted is not None:
            return {
                "success": True,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks),
                "error": None,
                "submitted": submitted,
            }

        if last_error is None:
            return {
                "success": True,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks),
                "error": None,
                "submitted": None,
            }

        return {
            "success": False,
            "stdout": "".join(stdout_chunks),
            "stderr": "".join(stderr_chunks),
            "error": last_error,
            "submitted": None,
        }

    async def _execute_with_stabilizer(
        self,
        code: str,
        input_vars: dict,
        tool_callables: dict,
    ) -> Optional[dict]:
        """Attempt to recover from syntax errors with sanitization."""
        sanitized = self._sanitize_code(code)
        if sanitized and sanitized != code:
            result = await self.interpreter.execute(
                code=sanitized,
                variables=input_vars,
                tools=tool_callables,
            )
            if result.get("success") or result.get("submitted"):
                return result

        if sanitized and self._can_execute_line_by_line(sanitized):
            return await self._execute_line_by_line(
                sanitized,
                input_vars,
                tool_callables,
            )

        return None

    def _format_validation_error(self, error: Exception) -> str:
        """Format validation errors for LLM feedback."""
        if isinstance(error, ValidationError):
            details = []
            for item in error.errors():
                loc = ".".join(str(part) for part in item.get("loc", [])) or "root"
                details.append(f"{loc}: {item.get('msg')}")
            return "[Type Error] " + "; ".join(details)
        if isinstance(error, jsonschema.ValidationError):
            loc = ".".join(str(part) for part in error.path) or "root"
            return f"[Type Error] {loc}: {error.message}"
        return f"[Type Error] {error}"

    def _validate_and_parse_output(
        self,
        submitted: dict,
    ) -> tuple[Optional[dict], Optional[str]]:
        """Validate and parse submitted output.

        Args:
            submitted: Raw submitted values from SUBMIT().

        Returns:
            Tuple of (parsed_outputs, error_message).
            If successful, error_message is None.
            If failed, parsed_outputs is None.
        """
        # Validate raw_output is a dict
        if not isinstance(submitted, dict):
            return None, (
                f"[Error] SUBMIT returned {type(submitted).__name__}, "
                f"expected dict with fields: {self.output_fields}"
            )

        # Validate all required output fields are present
        missing = set(self.required_fields) - set(submitted.keys())
        if missing:
            return None, (
                f"[Error] Missing output fields: {sorted(missing)}. "
                f"Use SUBMIT({', '.join(self.required_fields)})"
            )

        if self.data_model is not None:
            try:
                model = self.data_model(**submitted)
                return model.get_json(), None
            except ValidationError as e:
                return None, self._format_validation_error(e)

        try:
            jsonschema.validate(instance=submitted, schema=self.output_schema)
        except jsonschema.ValidationError as e:
            return None, self._format_validation_error(e)

        return submitted, None

    async def call(self, inputs, training: bool = False):
        """Execute the REPL loop.

        Args:
            inputs: Input data model.
            training: Whether in training mode.

        Returns:
            JsonDataModel with the computed output.
        """
        if not inputs:
            return None

        # Initialize interpreter
        async with self.interpreter:
            history = REPLHistory()
            llm_call_count = 0

            # Convert inputs to dict for variable injection
            if hasattr(inputs, "get_json"):
                input_vars = inputs.get_json()
            else:
                input_vars = dict(inputs)

            # Build REPLVariables with rich metadata
            variables = self._build_variables(inputs)

            # Build tool callables (sync wrappers around async tools)
            tool_callables = self._build_tool_callables()

            def _reserve_llm_calls(count: int) -> bool:
                """Reserve LLM calls without double-counting."""
                nonlocal llm_call_count
                if llm_call_count + count > self.max_llm_calls:
                    return False
                llm_call_count += count
                return True

            async def _llm_query(prompt: str) -> str:
                """Query the sub-LLM with a prompt (no call counting)."""
                try:
                    messages = ChatMessages(
                        messages=[ChatMessage(role="user", content=prompt)]
                    )
                    result = await ops.predict(
                        messages,
                        language_model=self.sub_language_model,
                        name="llm_query",
                    )
                    if result:
                        content = result.get("content", "")
                        if isinstance(content, str):
                            return content
                        return str(content)
                    return ""
                except Exception as e:
                    return f"ERROR: {str(e)}"

            # Create llm_query as async (interpreter handles async calls)
            async def llm_query(prompt: str) -> str:
                """Query the sub-LLM with a prompt."""
                if not _reserve_llm_calls(1):
                    return "ERROR: Maximum LLM calls exceeded"
                return await _llm_query(prompt)

            # Create batched query function
            async def llm_query_batched(prompts: List[str]) -> List[str]:
                """Query the sub-LLM with multiple prompts concurrently."""
                if not prompts:
                    return []
                if not _reserve_llm_calls(len(prompts)):
                    return ["ERROR: Maximum LLM calls exceeded"] * len(prompts)

                tasks = [_llm_query(p) for p in prompts]
                return await asyncio.gather(*tasks)

            # Add LLM functions to tools
            tool_callables["llm_query"] = llm_query
            tool_callables["llm_query_batched"] = llm_query_batched

            # Main REPL loop
            submitted_output = None

            for iteration in range(self.max_iterations):
                # Build context for generator with iteration info
                context = self._build_context(
                    variables, history, iteration
                )

                # Generate next action
                action = await self.generator(context, training=training)

                if not action:
                    break

                action_json = action.get_json() if isinstance(action, JsonDataModel) else action
                if not isinstance(action_json, dict):
                    break

                # Allow direct output submissions (schema match) without REPL code
                if (
                    "code" not in action_json
                    and "code_lines" not in action_json
                    and "reasoning" not in action_json
                ):
                    parsed, error = self._validate_and_parse_output(action_json)
                    if error:
                        history = history.append(
                            iteration=iteration,
                            reasoning="",
                            code="<DIRECT_OUTPUT>",
                            stdout="",
                            error=error,
                        )
                        continue

                    submitted_output = parsed
                    history = history.append(
                        iteration=iteration,
                        reasoning="",
                        code="<DIRECT_OUTPUT>",
                        stdout=f"SUBMIT: {parsed}",
                    )
                    break

                reasoning = action_json.get("reasoning", "")
                raw_code = action_json.get("code", "")
                if not raw_code and isinstance(action_json.get("code_lines"), list):
                    raw_code = "\n".join(
                        line for line in action_json.get("code_lines") if line is not None
                    )

                # Strip markdown code fences from LLM output
                code = strip_code_fences(raw_code)

                # Execute code
                result = await self.interpreter.execute(
                    code=code,
                    variables=input_vars,
                    tools=tool_callables,
                )
                if not result.get("success") and "SyntaxError" in (
                    result.get("error") or ""
                ):
                    stabilized = await self._execute_with_stabilizer(
                        code,
                        input_vars,
                        tool_callables,
                    )
                    if stabilized is not None:
                        result = stabilized

                # Check for SUBMIT
                if result.get("submitted"):
                    # Validate the output
                    parsed, error = self._validate_and_parse_output(
                        result["submitted"]
                    )

                    if error:
                        # Give feedback to LLM about validation error
                        history = history.append(
                            iteration=iteration,
                            reasoning=reasoning,
                            code=code,
                            stdout=result.get("stdout", ""),
                            error=error,
                        )
                        continue  # Let LLM try again

                    submitted_output = parsed
                    # Record successful submission in history
                    history = history.append(
                        iteration=iteration,
                        reasoning=reasoning,
                        code=code,
                        stdout=f"SUBMIT: {parsed}",
                    )
                    break

                # Record history (immutable - returns new instance)
                history = history.append(
                    iteration=iteration,
                    reasoning=reasoning,
                    code=code,
                    stdout=result.get("stdout", ""),
                    error=result.get("error"),
                )

            # Build final output
            if submitted_output:
                output_json = submitted_output
            else:
                # Fallback: extract from history
                extraction_context = self._build_extraction_context(
                    variables, history
                )
                extracted = await self.extractor(extraction_context, training=training)
                if extracted:
                    parsed, error = self._validate_and_parse_output(
                        extracted.get_json()
                    )
                    if error:
                        raise ValueError(
                            f"Fallback extraction failed validation: {error}"
                        )
                    output_json = parsed
                else:
                    output_json = {}

            # Optionally include history (using to_trajectory for clean output)
            if self.return_history:
                output_json["_history"] = history.to_trajectory()

            return JsonDataModel(
                json=output_json,
                schema=self.output_schema,
                name=f"{self.name}_output",
            )

    def _build_tool_callables(self) -> Dict[str, Callable]:
        """Convert Tool modules to async callables.

        Returns:
            Dict of async callables that wrap Synalinks Tool modules.
        """
        callables: Dict[str, Callable] = {}

        for name, tool in self.tools.items():
            # Create a closure to properly capture the tool
            def make_async_caller(t: Tool) -> Callable:
                async def async_caller(**kwargs: Any) -> Any:
                    result = await t(**kwargs)
                    if hasattr(result, "get_json"):
                        return result.get_json()
                    return result

                return async_caller

            callables[name] = make_async_caller(tool)

        return callables

    def _build_context(
        self,
        variables: List[REPLVariable],
        history: REPLHistory,
        iteration: int,
    ) -> JsonDataModel:
        """Build context for the generator with iteration info.

        Args:
            variables: List of REPLVariable instances.
            history: Current execution history.
            iteration: Current iteration number (0-indexed).

        Returns:
            JsonDataModel with context for the generator.
        """
        # Format iteration as "current/max" (1-indexed for display)
        iteration_info = f"{iteration + 1}/{self.max_iterations}"

        context = {
            "variables_info": self._format_variables_info(variables),
            "repl_history": history.format_for_prompt(),
            "iteration": iteration_info,
        }

        if self.tools:
            context["available_tools"] = self._format_tool_descriptions()

        return JsonDataModel(
            json=context,
            schema={"type": "object"},
            name="repl_context",
        )

    def _build_extraction_context(
        self,
        variables: List[REPLVariable],
        history: REPLHistory,
    ) -> JsonDataModel:
        """Build context for fallback extraction.

        Args:
            variables: List of REPLVariable instances.
            history: Complete execution history.

        Returns:
            JsonDataModel with extraction context.
        """
        context = {
            "variables_info": self._format_variables_info(variables),
            "repl_history": history.format_for_prompt(),
            "task": (
                "Extract the final answer from the execution history above. "
                f"Required output fields: {', '.join(self.required_fields)}"
            ),
        }
        return JsonDataModel(
            json=context,
            schema={"type": "object"},
            name="extraction_context",
        )

    async def compute_output_spec(self, inputs, training=False):
        """Compute output specification."""
        return SymbolicDataModel(
            schema=self.output_schema,
            name=self.name,
        )

    def get_config(self):
        """Get configuration for serialization."""
        config = {
            "schema": self.output_schema,
            "max_iterations": self.max_iterations,
            "max_llm_calls": self.max_llm_calls,
            "max_output_chars": self.max_output_chars,
            "max_tokens": self.max_tokens,
            "return_history": self.return_history,
            "instructions": self._instructions,
            "seed_instructions": self._seed_instructions,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }

        # Serialize language models
        config["language_model"] = serialization_lib.serialize_synalinks_object(
            self.language_model
        )
        if self.sub_language_model != self.language_model:
            config["sub_language_model"] = serialization_lib.serialize_synalinks_object(
                self.sub_language_model
            )
        else:
            config["sub_language_model"] = None

        # Serialize tools
        config["tools"] = [
            serialization_lib.serialize_synalinks_object(tool)
            for tool in self.tools.values()
        ]

        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration dict."""
        # Deserialize language models
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )

        sub_language_model_config = config.pop("sub_language_model", None)
        if sub_language_model_config:
            sub_language_model = serialization_lib.deserialize_synalinks_object(
                sub_language_model_config
            )
        else:
            sub_language_model = None

        # Deserialize tools
        tools_config = config.pop("tools", [])
        tools = [
            serialization_lib.deserialize_synalinks_object(tool_config)
            for tool_config in tools_config
        ]

        return cls(
            language_model=language_model,
            sub_language_model=sub_language_model,
            tools=tools if tools else None,
            **config,
        )


# Backward-compatible alias - REPLModule is deprecated, use RLM instead
REPLModule = RLM
