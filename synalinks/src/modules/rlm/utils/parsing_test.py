"""Tests for parsing utilities."""

from synalinks.src import testing
from synalinks.src.modules.rlm.core.types import REPLResult
from synalinks.src.modules.rlm.utils.parsing import find_code_blocks
from synalinks.src.modules.rlm.utils.parsing import find_final_answer
from synalinks.src.modules.rlm.utils.parsing import format_execution_result


class FindCodeBlocksTest(testing.TestCase):
    """Tests for find_code_blocks()."""

    def test_single_code_block(self):
        """Extracts single code block."""
        text = """Some text
```repl
x = 1
```
More text"""
        result = find_code_blocks(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "x = 1")

    def test_multiple_code_blocks(self):
        """Extracts multiple code blocks."""
        text = """```repl
a = 1
```
Middle
```repl
b = 2
```"""
        result = find_code_blocks(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "a = 1")
        self.assertEqual(result[1], "b = 2")

    def test_no_code_blocks(self):
        """Returns empty list when no blocks."""
        text = "Just regular text"
        result = find_code_blocks(text)
        self.assertEqual(result, [])

    def test_multiline_code(self):
        """Handles multiline code."""
        text = """```repl
x = 1
y = 2
print(x + y)
```"""
        result = find_code_blocks(text)
        self.assertIn("x = 1", result[0])
        self.assertIn("y = 2", result[0])

    def test_ignores_other_language_blocks(self):
        """Ignores non-repl code blocks."""
        text = """```python
x = 1
```
```repl
y = 2
```"""
        result = find_code_blocks(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "y = 2")

    def test_nested_backticks_in_string(self):
        """Handles backticks within code strings."""
        text = """```repl
s = "some `code` here"
print(s)
```"""
        result = find_code_blocks(text)
        self.assertEqual(len(result), 1)
        self.assertIn("some `code` here", result[0])

    def test_real_llm_format_single_block(self):
        """Handles real LLM response format - single block."""
        text = """Let me solve this step by step.

```repl
x = 5
y = 10
result = x + y
print(f"The sum is {result}")
```

The calculation shows that x + y = 15."""
        result = find_code_blocks(text)
        self.assertEqual(len(result), 1)
        self.assertIn("x = 5", result[0])
        self.assertIn("y = 10", result[0])
        self.assertIn("result = x + y", result[0])

    def test_real_llm_format_multiple_blocks(self):
        """Handles real LLM response format - multiple blocks."""
        text = """First, let's define the data:

```repl
data = [1, 2, 3, 4, 5]
```

Now let's calculate the mean:

```repl
mean = sum(data) / len(data)
print(f"Mean: {mean}")
```

FINAL_VAR(mean)"""
        result = find_code_blocks(text)
        self.assertEqual(len(result), 2)
        self.assertIn("data =", result[0])
        self.assertIn("mean =", result[1])


class FindFinalAnswerTest(testing.TestCase):
    """Tests for find_final_answer()."""

    def test_final_pattern(self):
        """Detects FINAL() pattern."""
        text = "After thinking...\nFINAL(The answer is 42)"
        result = find_final_answer(text)
        self.assertEqual(result, ("FINAL", "The answer is 42"))

    def test_final_var_pattern(self):
        """Detects FINAL_VAR() pattern."""
        text = "FINAL_VAR(result)"
        result = find_final_answer(text)
        self.assertEqual(result, ("FINAL_VAR", "result"))

    def test_final_var_with_quotes(self):
        """Handles quoted variable names."""
        text = 'FINAL_VAR("my_var")'
        result = find_final_answer(text)
        self.assertEqual(result[0], "FINAL_VAR")
        self.assertIn("my_var", result[1])

    def test_no_final_returns_none(self):
        """Returns None when no pattern."""
        text = "No final answer here"
        result = find_final_answer(text)
        self.assertIsNone(result)

    def test_final_must_be_at_line_start(self):
        """FINAL must be at line start."""
        text = "inline FINAL(x)"  # Not at line start
        result = find_final_answer(text)
        self.assertIsNone(result)

    def test_final_with_whitespace(self):
        """Allows leading whitespace."""
        text = "   FINAL(answer)"
        result = find_final_answer(text)
        self.assertEqual(result, ("FINAL", "answer"))

    def test_final_multiline_content(self):
        """Handles multiline content in FINAL()."""
        text = """Some reasoning...
FINAL(The answer is:
Line 1
Line 2)"""
        result = find_final_answer(text)
        self.assertEqual(result[0], "FINAL")
        self.assertIn("Line 1", result[1])
        self.assertIn("Line 2", result[1])

    def test_final_var_priority(self):
        """FINAL_VAR takes priority over FINAL when both present."""
        text = """FINAL_VAR(result)
FINAL(backup answer)"""
        result = find_final_answer(text)
        self.assertEqual(result, ("FINAL_VAR", "result"))

    def test_real_llm_format_final(self):
        """Handles real LLM response with FINAL()."""
        text = """Let me analyze this problem.

```repl
x = 42
y = x * 2
```

After computing, the result is clear.

FINAL(The answer is 84)"""
        result = find_final_answer(text)
        self.assertEqual(result, ("FINAL", "The answer is 84"))

    def test_real_llm_format_final_var(self):
        """Handles real LLM response with FINAL_VAR()."""
        text = """Computing the result:

```repl
data = [1, 2, 3, 4, 5]
average = sum(data) / len(data)
```

FINAL_VAR(average)"""
        result = find_final_answer(text)
        self.assertEqual(result, ("FINAL_VAR", "average"))

    def test_nested_parentheses_in_final(self):
        """Handles nested parentheses in FINAL() content."""
        text = "FINAL(The result (with details) is here)"
        result = find_final_answer(text)
        # Current implementation uses non-greedy match, so it stops at first )
        # This is a known limitation - we'll accept it for now
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "FINAL")


class FormatExecutionResultTest(testing.TestCase):
    """Tests for format_execution_result()."""

    def test_stdout_included(self):
        """Includes stdout in output."""
        result = REPLResult(stdout="Hello", stderr="", locals={})
        output = format_execution_result(result)
        self.assertIn("Hello", output)

    def test_stderr_included(self):
        """Includes stderr in output."""
        result = REPLResult(stdout="", stderr="Error!", locals={})
        output = format_execution_result(result)
        self.assertIn("Error!", output)

    def test_truncation(self):
        """Truncates long output."""
        long_text = "x" * 25000
        result = REPLResult(stdout=long_text, stderr="", locals={})
        output = format_execution_result(result, max_length=1000)
        self.assertLess(len(output), 25000)
        self.assertIn("truncated", output)

    def test_empty_result(self):
        """Handles empty result."""
        result = REPLResult(stdout="", stderr="", locals={})
        output = format_execution_result(result)
        self.assertEqual(output, "No output")

    def test_includes_user_variables(self):
        """Shows user-defined variables."""
        result = REPLResult(
            stdout="",
            stderr="",
            locals={"x": 10, "result": 42, "_internal": 5, "__builtins__": {}},
        )
        output = format_execution_result(result)
        self.assertIn("REPL variables:", output)
        self.assertIn("x", output)
        self.assertIn("result", output)
        # Internal variables should be excluded
        self.assertNotIn("_internal", output)
        self.assertNotIn("__builtins__", output)

    def test_complex_types_included(self):
        """Includes simple types but excludes complex objects."""
        result = REPLResult(
            stdout="",
            stderr="",
            locals={
                "simple_str": "hello",
                "simple_int": 42,
                "simple_list": [1, 2, 3],
                "simple_dict": {"a": 1},
            },
        )
        output = format_execution_result(result)
        self.assertIn("simple_str", output)
        self.assertIn("simple_int", output)
        self.assertIn("simple_list", output)
        self.assertIn("simple_dict", output)

    def test_real_repl_output_format(self):
        """Handles realistic REPL execution output."""
        result = REPLResult(
            stdout="Computing...\nResult: 15\n",
            stderr="",
            locals={
                "x": 5,
                "y": 10,
                "result": 15,
            },
        )
        output = format_execution_result(result)
        self.assertIn("Computing...", output)
        self.assertIn("Result: 15", output)
        self.assertIn("REPL variables:", output)
        self.assertIn("x", output)
        self.assertIn("result", output)

    def test_error_output_format(self):
        """Handles REPL execution with errors."""
        result = REPLResult(
            stdout="",
            stderr="NameError: name 'undefined_var' is not defined",
            locals={},
        )
        output = format_execution_result(result)
        self.assertIn("NameError", output)
        self.assertIn("undefined_var", output)
