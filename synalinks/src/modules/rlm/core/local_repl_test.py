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

    def test_default_sub_model_parameter_accepted(self):
        """LocalREPL accepts default_sub_model parameter."""
        repl = LocalREPL(default_sub_model="sub")
        self.assertEqual(repl.default_sub_model, "sub")

    def test_default_sub_model_defaults_to_none(self):
        """default_sub_model defaults to None when not provided."""
        repl = LocalREPL()
        self.assertIsNone(repl.default_sub_model)

    def test_default_sub_model_routing_indication(self):
        """default_sub_model parameter indicates routing to cheaper model."""
        # This parameter is documentation for how llm_query should route
        # The actual routing is done by LMHandler.create_llm_query_fn()
        repl = LocalREPL(default_sub_model="groq/openai/gpt-oss-20b")
        self.assertEqual(repl.default_sub_model, "groq/openai/gpt-oss-20b")

    def test_reset_with_new_default_sub_model(self):
        """Reset can update default_sub_model."""
        repl = LocalREPL(default_sub_model="sub1")
        self.assertEqual(repl.default_sub_model, "sub1")

        repl.reset(default_sub_model="sub2")
        self.assertEqual(repl.default_sub_model, "sub2")

    def test_reset_preserves_default_sub_model_if_not_provided(self):
        """Reset preserves default_sub_model if not explicitly changed."""
        repl = LocalREPL(default_sub_model="sub")
        self.assertEqual(repl.default_sub_model, "sub")

        repl.reset()
        self.assertEqual(repl.default_sub_model, "sub")

    def test_llm_query_batched_injection(self):
        """llm_query_batched function can be injected."""
        called_prompts = []

        def mock_llm_query_batched(prompts):
            called_prompts.extend(prompts)
            return [f"response_{i}" for i in range(len(prompts))]

        repl = LocalREPL(llm_query_batched_fn=mock_llm_query_batched)
        result = repl.execute("results = llm_query_batched(['Q1', 'Q2', 'Q3'])")

        self.assertTrue(result.success)
        self.assertEqual(
            result.locals["results"], ["response_0", "response_1", "response_2"]
        )
        self.assertEqual(called_prompts, ["Q1", "Q2", "Q3"])

    def test_llm_query_batched_not_available_by_default(self):
        """llm_query_batched is not available when not injected."""
        repl = LocalREPL()
        result = repl.execute("llm_query_batched(['test'])")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.exception)

    def test_reset_with_new_llm_query_batched(self):
        """Reset can update llm_query_batched function."""

        def mock_batched_1(prompts):
            return ["response_1"] * len(prompts)

        def mock_batched_2(prompts):
            return ["response_2"] * len(prompts)

        repl = LocalREPL(llm_query_batched_fn=mock_batched_1)
        result1 = repl.execute("r = llm_query_batched(['test'])")
        self.assertEqual(result1.locals["r"], ["response_1"])

        repl.reset(llm_query_batched_fn=mock_batched_2)
        result2 = repl.execute("r = llm_query_batched(['test'])")
        self.assertEqual(result2.locals["r"], ["response_2"])
