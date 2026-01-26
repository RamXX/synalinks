# License Apache 2.0: (c) 2025 Synalinks Team

"""Data models for REPL execution history and state.

These types represent the state and history of REPL-based execution:
- REPLVariable: Metadata about variables available in the REPL
- REPLEntry: A single interaction (reasoning, code, output)
- REPLHistory: Container for the full interaction history (immutable)
"""

import json
from typing import Any, Iterator, List, Optional

from pydantic import ConfigDict, Field

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.pydantic.core import DataModel


@synalinks_export("synalinks.REPLVariable")
class REPLVariable(DataModel):
    """Metadata about a variable available in the REPL environment.

    Provides structured context to the LLM about each input variable,
    including type information, descriptions, constraints, and a preview.

    Attributes:
        name: Variable name (used to access it in code).
        type_name: Python type name of the value.
        desc: Description from field metadata, if available.
        constraints: Constraints from field metadata, if available.
        total_length: Total character length of the serialized value.
        preview: Truncated preview of the value.

    Example:

    ```python
    variable = REPLVariable.from_value(
        name="document",
        value="This is a very long document...",
        preview_chars=100,
    )
    print(variable.format())
    # Variable: `document` (access it in your code)
    # Type: str
    # Total length: 1,234 characters
    # Preview:
    # ```
    # This is a very long document...
    # ```
    ```
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Variable name")
    type_name: str = Field(description="Python type name")
    desc: str = Field(default="", description="Description from field metadata")
    constraints: str = Field(default="", description="Constraints from field metadata")
    total_length: int = Field(description="Total character length of serialized value")
    preview: str = Field(description="Truncated preview of the value")

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        field_info: Optional[Any] = None,
        preview_chars: int = 500,
    ) -> "REPLVariable":
        """Create REPLVariable from an actual value and optional field info.

        Args:
            name: Variable name.
            value: The actual value.
            field_info: Optional pydantic FieldInfo with desc/constraints metadata.
            preview_chars: Maximum characters for preview.

        Returns:
            REPLVariable instance with computed metadata.
        """
        # Serialize value for display
        if isinstance(value, (dict, list)):
            try:
                value_str = json.dumps(value, indent=2, default=str)
            except (TypeError, ValueError):
                value_str = str(value)
        else:
            value_str = str(value)

        is_truncated = len(value_str) > preview_chars
        preview = value_str[:preview_chars] + ("..." if is_truncated else "")

        # Extract desc and constraints from field_info if provided
        desc = ""
        constraints = ""
        if field_info is not None:
            # Handle pydantic FieldInfo
            if hasattr(field_info, "description") and field_info.description:
                raw_desc = field_info.description
                # Skip placeholder descs like "${name}"
                if not raw_desc.startswith("${"):
                    desc = raw_desc
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if isinstance(extra, dict):
                    if "desc" in extra and not extra["desc"].startswith("${"):
                        desc = desc or extra["desc"]
                    constraints = extra.get("constraints", "")

        return cls(
            name=name,
            type_name=type(value).__name__,
            desc=desc,
            constraints=constraints,
            total_length=len(value_str),
            preview=preview,
        )

    def format(self) -> str:
        """Format variable metadata for prompt inclusion.

        Returns:
            Formatted string suitable for including in an LLM prompt.
        """
        lines = [f"Variable: `{self.name}` (access it in your code)"]
        lines.append(f"Type: {self.type_name}")
        if self.desc:
            lines.append(f"Description: {self.desc}")
        if self.constraints:
            lines.append(f"Constraints: {self.constraints}")
        lines.append(f"Total length: {self.total_length:,} characters")
        lines.append(f"Preview:\n```\n{self.preview}\n```")
        return "\n".join(lines)


@synalinks_export("synalinks.REPLEntry")
class REPLEntry(DataModel):
    """A single REPL iteration entry (immutable).

    Captures the reasoning, code, and output from one iteration
    of the REPL loop.

    Attributes:
        iteration: The iteration number (0-indexed).
        reasoning: LLM's reasoning about what to do next.
        code: Python code that was executed.
        stdout: Captured stdout from execution.
        error: Error message if execution failed, None otherwise.
    """

    model_config = ConfigDict(frozen=True)

    iteration: int = Field(description="Iteration number (0-indexed)")
    reasoning: str = Field(description="LLM's reasoning about next action")
    code: str = Field(description="Python code that was executed")
    stdout: str = Field(default="", description="Captured stdout from execution")
    error: Optional[str] = Field(default=None, description="Error message if any")

    def format(self, max_output_chars: int = 5000) -> str:
        """Format this entry for inclusion in prompts.

        Args:
            max_output_chars: Maximum characters for output display.

        Returns:
            Formatted string representation.
        """
        output = self.stdout if not self.error else f"[Error] {self.error}"
        if len(output) > max_output_chars:
            output = (
                output[:max_output_chars]
                + f"\n... (truncated to {max_output_chars}/{len(output):,} chars)"
            )

        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        return (
            f"=== Step {self.iteration + 1} ===\n"
            f"{reasoning_line}"
            f"Code:\n```python\n{self.code}\n```\n"
            f"Output ({len(self.stdout):,} chars):\n{output}"
        )


@synalinks_export("synalinks.REPLHistory")
class REPLHistory(DataModel):
    """Complete REPL execution history (immutable).

    Maintains a list of REPL entries representing the full
    execution trajectory of an RLM session.

    This class is immutable: append() returns a new instance
    with the entry added, preserving the original.

    Attributes:
        entries: List of REPL iteration entries.

    Example:

    ```python
    history = REPLHistory()

    # Append returns a NEW history instance
    history = history.append(
        iteration=0,
        reasoning="First, let me explore the data structure",
        code="print(type(data))",
        stdout="<class 'dict'>",
    )

    prompt_context = history.format_for_prompt()
    ```
    """

    model_config = ConfigDict(frozen=True)

    entries: List[REPLEntry] = Field(
        default_factory=list, description="List of REPL iterations"
    )

    def append(
        self,
        *,
        iteration: int,
        reasoning: str,
        code: str,
        stdout: str = "",
        error: Optional[str] = None,
    ) -> "REPLHistory":
        """Append a new entry and return a NEW REPLHistory instance.

        Args:
            iteration: Iteration number (0-indexed).
            reasoning: LLM's reasoning for this iteration.
            code: Python code that was executed.
            stdout: Captured stdout from execution.
            error: Error message if execution failed.

        Returns:
            New REPLHistory instance with the entry appended.
        """
        new_entry = REPLEntry(
            iteration=iteration,
            reasoning=reasoning,
            code=code,
            stdout=stdout,
            error=error,
        )
        return REPLHistory(entries=list(self.entries) + [new_entry])

    def format_for_prompt(self, max_output_chars: int = 5000) -> str:
        """Format history for LLM context.

        Args:
            max_output_chars: Maximum characters per entry output.

        Returns:
            Formatted string representation of the execution history
            suitable for including in an LLM prompt.
        """
        if not self.entries:
            return "You have not interacted with the REPL environment yet."

        return "\n\n".join(
            entry.format(max_output_chars=max_output_chars) for entry in self.entries
        )

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)

    def __iter__(self) -> Iterator[REPLEntry]:
        """Iterate over entries."""
        return iter(self.entries)

    def __bool__(self) -> bool:
        """Return True if history has entries."""
        return len(self.entries) > 0

    def get_last_entry(self) -> Optional[REPLEntry]:
        """Get the most recent entry.

        Returns:
            The last entry, or None if history is empty.
        """
        return self.entries[-1] if self.entries else None

    def had_errors(self) -> bool:
        """Check if any iteration had errors.

        Returns:
            True if any entry has a non-None error.
        """
        return any(entry.error is not None for entry in self.entries)

    def get_total_output_chars(self) -> int:
        """Get total characters of stdout across all entries.

        Returns:
            Total character count.
        """
        return sum(len(entry.stdout) for entry in self.entries)

    def to_trajectory(self) -> List[dict]:
        """Convert history to a list of dicts for debugging output.

        Returns:
            List of entry dictionaries.
        """
        return [
            {
                "iteration": e.iteration,
                "reasoning": e.reasoning,
                "code": e.code,
                "stdout": e.stdout,
                "error": e.error,
            }
            for e in self.entries
        ]
