"""Parsing utilities for RLM response processing."""

import re
from typing import Optional

from synalinks.src.modules.rlm.core.types import REPLResult


def find_code_blocks(text: str) -> list[str]:
    """Extract REPL code blocks from LLM response.

    Finds all code blocks wrapped in triple backticks with 'repl' language identifier.

    Args:
        text: LLM response text

    Returns:
        List of code strings (without backticks wrapper)

    Example:
        >>> text = '''Let me analyze this.
        ... ```repl
        ... x = 1 + 2
        ... print(x)
        ... ```
        ... The answer is 3.'''
        >>> find_code_blocks(text)
        ['x = 1 + 2\\nprint(x)']
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def find_final_answer(text: str) -> Optional[tuple[str, str]]:
    """Detect FINAL() or FINAL_VAR() termination pattern.

    Searches for patterns at the start of a line:
    - FINAL(content) - direct answer
    - FINAL_VAR(variable_name) - reference to REPL variable

    Args:
        text: LLM response text

    Returns:
        Tuple of (type, content) where type is "FINAL" or "FINAL_VAR",
        or None if no pattern found

    Example:
        >>> find_final_answer("After analysis, FINAL_VAR(result)")
        ('FINAL_VAR', 'result')
        >>> find_final_answer("FINAL(The answer is 42)")
        ('FINAL', 'The answer is 42')
        >>> find_final_answer("No final answer here")
        None
    """
    # Check FINAL_VAR first (more specific)
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ("FINAL_VAR", match.group(1).strip())

    # Check FINAL pattern
    final_pattern = r"^\s*FINAL\((.*?)\)"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ("FINAL", match.group(1).strip())

    return None


def format_execution_result(result: REPLResult, max_length: int = 20000) -> str:
    """Format REPL execution result for inclusion in next prompt.

    Combines stdout, stderr, and variable information into a string.
    Truncates if exceeds max_length.

    Args:
        result: REPLResult from code execution
        max_length: Maximum character length before truncation

    Returns:
        Formatted string for prompt
    """
    parts = []

    if result.stdout:
        parts.append(result.stdout)

    if result.stderr:
        parts.append(result.stderr)

    # Show key variables (exclude internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        parts.append(f"REPL variables: {list(important_vars.keys())}")

    output = "\n\n".join(parts) if parts else "No output"

    if len(output) > max_length:
        output = (
            output[:max_length] + f"\n... [{len(output) - max_length} chars truncated]"
        )

    return output
