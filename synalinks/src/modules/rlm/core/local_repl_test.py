"""Tests for LocalREPL."""

import json
import os
import time

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

    def test_final_var_function_available(self):
        """FINAL_VAR function is available in builtins."""
        repl = LocalREPL()
        result = repl.execute("answer = FINAL_VAR(42)")

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, 42)
        self.assertEqual(result.locals["answer"], 42)

    def test_final_var_sets_final_answer(self):
        """FINAL_VAR sets final_answer field in REPLResult."""
        repl = LocalREPL()
        result = repl.execute("FINAL_VAR({'result': 'success', 'value': 100})")

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, {"result": "success", "value": 100})

    def test_final_var_with_structured_data(self):
        """FINAL_VAR works with complex structured data."""
        repl = LocalREPL()
        code = """
data = {'name': 'test', 'scores': [1, 2, 3], 'nested': {'key': 'value'}}
FINAL_VAR(data)
"""
        result = repl.execute(code)

        self.assertTrue(result.success)
        self.assertEqual(
            result.final_answer,
            {"name": "test", "scores": [1, 2, 3], "nested": {"key": "value"}},
        )

    def test_final_var_resets_on_each_execute(self):
        """final_answer resets to None on each execute."""
        repl = LocalREPL()

        result1 = repl.execute("FINAL_VAR(1)")
        self.assertEqual(result1.final_answer, 1)

        result2 = repl.execute("x = 2")
        self.assertIsNone(result2.final_answer)

    def test_timeout_parameter_accepted(self):
        """LocalREPL accepts timeout parameter."""
        repl = LocalREPL(timeout=5.0)
        self.assertEqual(repl.timeout, 5.0)

    def test_timeout_defaults_to_none(self):
        """timeout defaults to None when not provided."""
        repl = LocalREPL()
        self.assertIsNone(repl.timeout)

    def test_timeout_enforced_on_long_execution(self):
        """Timeout interrupts long-running code."""
        repl = LocalREPL(timeout=0.1)
        result = repl.execute("import time; time.sleep(10)")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.exception)
        self.assertIsInstance(result.exception, TimeoutError)

    def test_timeout_per_execution_overrides_instance(self):
        """Per-execution timeout overrides instance timeout."""
        repl = LocalREPL(timeout=10.0)
        result = repl.execute("import time; time.sleep(5)", timeout=0.1)

        self.assertFalse(result.success)
        self.assertIsInstance(result.exception, TimeoutError)

    def test_timeout_allows_fast_execution(self):
        """Timeout does not interfere with fast code."""
        repl = LocalREPL(timeout=1.0)
        result = repl.execute("x = sum(range(100))")

        self.assertTrue(result.success)
        self.assertEqual(result.locals["x"], 4950)

    def test_reset_with_new_timeout(self):
        """Reset can update timeout."""
        repl = LocalREPL(timeout=1.0)
        self.assertEqual(repl.timeout, 1.0)

        repl.reset(timeout=5.0)
        self.assertEqual(repl.timeout, 5.0)

    def test_reset_preserves_timeout_if_not_provided(self):
        """Reset preserves timeout if not explicitly changed."""
        repl = LocalREPL(timeout=2.0)
        self.assertEqual(repl.timeout, 2.0)

        repl.reset()
        self.assertEqual(repl.timeout, 2.0)

    def test_llm_query_batched_auto_created_from_llm_query(self):
        """llm_query_batched auto-created when llm_query provided."""
        call_count = []

        def mock_llm_query(prompt):
            call_count.append(prompt)
            return f"response:{prompt}"

        repl = LocalREPL(llm_query_fn=mock_llm_query)
        result = repl.execute("results = llm_query_batched(['Q1', 'Q2', 'Q3'])")

        self.assertTrue(result.success)
        self.assertEqual(
            result.locals["results"], ["response:Q1", "response:Q2", "response:Q3"]
        )
        self.assertEqual(len(call_count), 3)

    def test_llm_query_batched_uses_asyncio_gather(self):
        """llm_query_batched executes calls concurrently."""
        call_times = []

        def slow_llm_query(prompt):
            call_times.append(time.time())
            time.sleep(0.1)
            return f"response:{prompt}"

        repl = LocalREPL(llm_query_fn=slow_llm_query)
        start = time.time()
        result = repl.execute("results = llm_query_batched(['Q1', 'Q2', 'Q3'])")
        elapsed = time.time() - start

        self.assertTrue(result.success)
        # Concurrent execution should take ~0.1s, not 0.3s
        self.assertLess(elapsed, 0.25)
        # All calls should start around the same time
        self.assertEqual(len(call_times), 3)

    def test_explicit_llm_query_batched_overrides_auto(self):
        """Explicitly provided llm_query_batched overrides auto-creation."""
        auto_called = []
        explicit_called = []

        def mock_llm_query(prompt):
            auto_called.append(prompt)
            return "auto"

        def mock_batched(prompts):
            explicit_called.extend(prompts)
            return ["explicit"] * len(prompts)

        repl = LocalREPL(llm_query_fn=mock_llm_query, llm_query_batched_fn=mock_batched)
        result = repl.execute("results = llm_query_batched(['Q1', 'Q2'])")

        self.assertTrue(result.success)
        self.assertEqual(result.locals["results"], ["explicit", "explicit"])
        self.assertEqual(explicit_called, ["Q1", "Q2"])
        self.assertEqual(auto_called, [])  # Auto version not used

    def test_load_context_with_dict(self):
        """load_context loads dict into namespace."""
        repl = LocalREPL()
        repl.load_context({"x": 42, "y": "hello", "z": [1, 2, 3]})

        self.assertEqual(repl.get_variable("x"), 42)
        self.assertEqual(repl.get_variable("y"), "hello")
        self.assertEqual(repl.get_variable("z"), [1, 2, 3])

    def test_load_context_with_json_string(self):
        """load_context parses JSON string into namespace."""
        repl = LocalREPL()
        json_str = json.dumps({"a": 100, "b": "world", "c": {"nested": True}})
        repl.load_context(json_str)

        self.assertEqual(repl.get_variable("a"), 100)
        self.assertEqual(repl.get_variable("b"), "world")
        self.assertEqual(repl.get_variable("c"), {"nested": True})

    def test_temp_dir_created_on_init(self):
        """Temp directory is created on initialization."""
        repl = LocalREPL()
        self.assertTrue(hasattr(repl, "temp_dir"))
        self.assertTrue(os.path.exists(repl.temp_dir))
        self.assertTrue(os.path.isdir(repl.temp_dir))
        # Cleanup
        repl.cleanup()

    def test_temp_dir_cleanup(self):
        """Temp directory is cleaned up properly."""
        repl = LocalREPL()
        temp_dir = repl.temp_dir
        self.assertTrue(os.path.exists(temp_dir))

        repl.cleanup()
        self.assertFalse(os.path.exists(temp_dir))
