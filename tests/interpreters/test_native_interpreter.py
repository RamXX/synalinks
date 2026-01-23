# License Apache 2.0: (c) 2025 Synalinks Team

"""Tests for NativePythonInterpreter."""

import pytest

from synalinks.src.interpreters.native import NativePythonInterpreter


class TestNativePythonInterpreter:
    """Tests for NativePythonInterpreter."""

    @pytest.fixture
    def interpreter(self):
        """Create a fresh interpreter instance."""
        return NativePythonInterpreter()

    @pytest.mark.asyncio
    async def test_basic_execution(self, interpreter):
        """Test basic code execution."""
        async with interpreter:
            result = await interpreter.execute(
                code="x = 1 + 1\nprint(x)",
                variables={},
                tools={},
            )

        assert result["success"] is True
        assert "2" in result["stdout"]
        assert result["error"] is None
        assert result["submitted"] is None

    @pytest.mark.asyncio
    async def test_variable_injection(self, interpreter):
        """Test that variables are properly injected."""
        async with interpreter:
            result = await interpreter.execute(
                code="result = x + y\nprint(result)",
                variables={"x": 10, "y": 20},
                tools={},
            )

        assert result["success"] is True
        assert "30" in result["stdout"]

    @pytest.mark.asyncio
    async def test_variable_persistence(self, interpreter):
        """Test that variables persist across executions."""
        async with interpreter:
            # First execution - define a variable
            result1 = await interpreter.execute(
                code="my_var = 42",
                variables={},
                tools={},
            )
            assert result1["success"] is True

            # Second execution - use the variable
            result2 = await interpreter.execute(
                code="print(my_var * 2)",
                variables={},
                tools={},
            )
            assert result2["success"] is True
            assert "84" in result2["stdout"]

    @pytest.mark.asyncio
    async def test_submit_function(self, interpreter):
        """Test SUBMIT function captures values and stops execution."""
        async with interpreter:
            result = await interpreter.execute(
                code='SUBMIT(answer="hello", score=0.9)',
                variables={},
                tools={},
            )

        assert result["success"] is True
        assert result["submitted"] is not None
        assert result["submitted"]["answer"] == "hello"
        assert result["submitted"]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_submit_stops_execution(self, interpreter):
        """Test that SUBMIT stops further code execution."""
        async with interpreter:
            result = await interpreter.execute(
                code='print("before")\nSUBMIT(value=1)\nprint("after")',
                variables={},
                tools={},
            )

        assert result["success"] is True
        assert "before" in result["stdout"]
        assert "after" not in result["stdout"]
        assert result["submitted"] == {"value": 1}

    @pytest.mark.asyncio
    async def test_error_handling(self, interpreter):
        """Test that errors are properly captured."""
        async with interpreter:
            result = await interpreter.execute(
                code="1 / 0",
                variables={},
                tools={},
            )

        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    @pytest.mark.asyncio
    async def test_syntax_error(self, interpreter):
        """Test that syntax errors are captured."""
        async with interpreter:
            result = await interpreter.execute(
                code="if True\n  print('missing colon')",
                variables={},
                tools={},
            )

        assert result["success"] is False
        assert "SyntaxError" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_injection(self, interpreter):
        """Test that tools are properly injected and callable."""
        def my_tool(x: int, y: int) -> int:
            return x * y

        async with interpreter:
            result = await interpreter.execute(
                code="result = my_tool(3, 4)\nprint(result)",
                variables={},
                tools={"my_tool": my_tool},
            )

        assert result["success"] is True
        assert "12" in result["stdout"]

    @pytest.mark.asyncio
    async def test_async_tool(self, interpreter):
        """Test that async tools work correctly."""
        async def async_tool(value: str) -> str:
            return f"processed: {value}"

        async with interpreter:
            result = await interpreter.execute(
                code='result = await async_tool("test")\nprint(result)',
                variables={},
                tools={"async_tool": async_tool},
            )

        assert result["success"] is True
        assert "processed: test" in result["stdout"]

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        """Test that output is truncated at max_output_chars."""
        interpreter = NativePythonInterpreter(max_output_chars=50)

        async with interpreter:
            result = await interpreter.execute(
                code="print('x' * 100)",
                variables={},
                tools={},
            )

        assert result["success"] is True
        assert len(result["stdout"]) <= 100  # Allow some margin for truncation message
        assert "truncated" in result["stdout"]

    @pytest.mark.asyncio
    async def test_builtin_restrictions(self, interpreter):
        """Test that dangerous builtins are not available."""
        async with interpreter:
            # __import__ should not be available
            result = await interpreter.execute(
                code="__import__('os')",
                variables={},
                tools={},
            )

        assert result["success"] is False
        # Either NameError (not defined) or TypeError (not callable)
        assert "Error" in result["error"]

    @pytest.mark.asyncio
    async def test_allowed_builtins(self, interpreter):
        """Test that allowed builtins are available."""
        async with interpreter:
            result = await interpreter.execute(
                code="print(len([1, 2, 3]))",
                variables={},
                tools={},
            )

        assert result["success"] is True
        assert "3" in result["stdout"]

    @pytest.mark.asyncio
    async def test_list_operations(self, interpreter):
        """Test list operations with builtins."""
        async with interpreter:
            result = await interpreter.execute(
                code="""
data = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"len: {len(data)}")
print(f"sum: {sum(data)}")
print(f"min: {min(data)}")
print(f"max: {max(data)}")
print(f"sorted: {sorted(data)}")
""",
                variables={},
                tools={},
            )

        assert result["success"] is True
        assert "len: 8" in result["stdout"]
        assert "sum: 31" in result["stdout"]
        assert "min: 1" in result["stdout"]
        assert "max: 9" in result["stdout"]

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, interpreter):
        """Test that input variables don't persist after execution."""
        async with interpreter:
            # Execute with an input variable
            result1 = await interpreter.execute(
                code="print(input_var)",
                variables={"input_var": "test_value"},
                tools={},
            )
            assert result1["success"] is True

            # The input variable should not persist
            result2 = await interpreter.execute(
                code="print(input_var)",
                variables={},
                tools={},
            )
            assert result2["success"] is False
            assert "NameError" in result2["error"]

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager protocol."""
        interpreter = NativePythonInterpreter()

        async with interpreter:
            assert interpreter._started is True

        assert interpreter._started is False

    @pytest.mark.asyncio
    async def test_get_namespace(self, interpreter):
        """Test namespace introspection."""
        async with interpreter:
            await interpreter.execute(
                code="my_var = 123",
                variables={},
                tools={},
            )

            namespace = interpreter.get_namespace()
            assert "my_var" in namespace
            assert namespace["my_var"] == 123

    @pytest.mark.asyncio
    async def test_clear_namespace(self, interpreter):
        """Test clearing the namespace."""
        async with interpreter:
            await interpreter.execute(
                code="my_var = 123",
                variables={},
                tools={},
            )

            interpreter.clear_namespace()
            namespace = interpreter.get_namespace()
            assert len(namespace) == 0


class TestNativePythonInterpreterThreadSafety:
    """Thread-safety tests for NativePythonInterpreter."""

    @pytest.mark.asyncio
    async def test_thread_ownership_set_on_first_execute(self):
        """Test that thread ownership is set on first execute."""
        import threading

        interpreter = NativePythonInterpreter()
        async with interpreter:
            assert interpreter._owner_thread is None

            await interpreter.execute(
                code="x = 1",
                variables={},
                tools={},
            )

            assert interpreter._owner_thread == threading.current_thread().ident

    @pytest.mark.asyncio
    async def test_thread_ownership_cleared_on_stop(self):
        """Test that thread ownership is cleared when interpreter stops."""
        interpreter = NativePythonInterpreter()

        async with interpreter:
            await interpreter.execute(code="x = 1", variables={}, tools={})
            assert interpreter._owner_thread is not None

        # After exiting context, ownership should be cleared
        assert interpreter._owner_thread is None

    @pytest.mark.asyncio
    async def test_different_thread_raises_error(self):
        """Test that using interpreter from different thread raises error."""
        import asyncio
        import concurrent.futures
        import threading

        interpreter = NativePythonInterpreter()

        # Start interpreter and establish ownership in current thread
        await interpreter.start()
        await interpreter.execute(code="x = 1", variables={}, tools={})

        current_thread = threading.current_thread().ident
        assert interpreter._owner_thread == current_thread

        # Try to use from a different thread
        error_raised = False
        error_message = ""

        def run_in_thread():
            nonlocal error_raised, error_message
            try:
                # Create new event loop for the thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        interpreter.execute(code="y = 2", variables={}, tools={})
                    )
                finally:
                    loop.close()
            except RuntimeError as e:
                error_raised = True
                error_message = str(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            future.result()

        await interpreter.stop()

        assert error_raised
        assert "not thread-safe" in error_message
        assert "cannot be shared" in error_message
