"""Tests for LocalREPL."""

from synalinks.src import testing
from synalinks.src.modules.rlm.core.local_repl import LocalREPL


class LocalREPLTest(testing.TestCase):
    """Tests for LocalREPL execution."""

    def test_instantiates_without_error(self):
        """LocalREPL instantiates successfully."""
        repl = LocalREPL()
        self.assertIsNotNone(repl)

    def test_execute_simple_expression(self):
        """Executes simple Python expression."""
        repl = LocalREPL()
        result = repl.execute("x = 1 + 2")

        self.assertTrue(result.success)
        self.assertEqual(result.locals["x"], 3)
        self.assertIsNone(result.exception)

    def test_execute_with_print(self):
        """Captures stdout from print statements."""
        repl = LocalREPL()
        result = repl.execute("print('hello')")

        self.assertTrue(result.success)
        self.assertIn("hello", result.stdout)

    def test_execute_with_error(self):
        """Captures exceptions from failed execution."""
        repl = LocalREPL()
        result = repl.execute("1 / 0")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.exception)
        self.assertIn("ZeroDivisionError", result.stderr)

    def test_safe_builtins_available(self):
        """Safe builtins are available."""
        repl = LocalREPL()
        result = repl.execute("x = sum([1, 2, 3])")

        self.assertTrue(result.success)
        self.assertEqual(result.locals["x"], 6)

    def test_dangerous_builtins_blocked(self):
        """Dangerous builtins like open are not available."""
        repl = LocalREPL()
        result = repl.execute("open('/tmp/test.txt')")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.exception)

    def test_get_variable(self):
        """Can retrieve variables from REPL state."""
        repl = LocalREPL()
        repl.execute("my_var = 42")

        value = repl.get_variable("my_var")
        self.assertEqual(value, 42)

    def test_get_nonexistent_variable_raises(self):
        """Getting nonexistent variable raises KeyError."""
        repl = LocalREPL()

        with self.assertRaises(KeyError):
            repl.get_variable("nonexistent")

    def test_reset_clears_state(self):
        """Reset clears REPL state."""
        repl = LocalREPL()
        repl.execute("x = 42")
        repl.reset()

        with self.assertRaises(KeyError):
            repl.get_variable("x")

    def test_llm_query_injection(self):
        """llm_query function can be injected."""
        called = []

        def mock_llm_query(prompt):
            called.append(prompt)
            return "mocked response"

        repl = LocalREPL(llm_query_fn=mock_llm_query)
        result = repl.execute("response = llm_query('test')")

        self.assertTrue(result.success)
        self.assertEqual(result.locals["response"], "mocked response")
        self.assertEqual(called, ["test"])
