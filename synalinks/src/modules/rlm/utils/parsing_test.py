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
