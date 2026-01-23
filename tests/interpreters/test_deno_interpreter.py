# License Apache 2.0: (c) 2025 Synalinks Team

"""Tests for DenoInterpreter.

These tests require Deno to be installed on the system.
Run: curl -fsSL https://deno.land/install.sh | sh
"""

import shutil

import pytest

from synalinks.src.interpreters.deno import DenoInterpreter

# Skip all tests if Deno is not installed
pytestmark = pytest.mark.skipif(
    shutil.which("deno") is None,
    reason="Deno is not installed. Install from: https://deno.land"
)


class TestDenoInterpreter:
    """Tests for DenoInterpreter."""

    @pytest.fixture
    def interpreter(self):
        """Create a fresh interpreter instance."""
        return DenoInterpreter()

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
        # SUBMIT raises FinalOutput, so "after" should not be printed
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
        assert result["error"] is not None
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
        assert result["error"] is not None
        assert "SyntaxError" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_injection(self, interpreter):
        """Test that tools are properly injected and callable."""
        def my_tool(x: int, y: int) -> dict:
            return {"result": x * y}

        async with interpreter:
            result = await interpreter.execute(
                code="result = my_tool(3, 4)\nprint(result['result'])",
                variables={},
                tools={"my_tool": my_tool},
            )

        assert result["success"] is True
        assert "12" in result["stdout"]

    @pytest.mark.asyncio
    async def test_async_tool(self, interpreter):
        """Test that async tools work correctly."""
        async def async_tool(value: str) -> dict:
            return {"message": f"processed: {value}"}

        async with interpreter:
            result = await interpreter.execute(
                code='result = async_tool("test")\nprint(result["message"])',
                variables={},
                tools={"async_tool": async_tool},
            )

        assert result["success"] is True
        assert "processed: test" in result["stdout"]

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        """Test that output is truncated at max_output_chars."""
        interpreter = DenoInterpreter(max_output_chars=50)

        async with interpreter:
            result = await interpreter.execute(
                code="print('x' * 100)",
                variables={},
                tools={},
            )

        assert result["success"] is True
        assert len(result["stdout"]) <= 100  # Allow margin for truncation msg
        assert "truncated" in result["stdout"]

    @pytest.mark.asyncio
    async def test_list_operations(self, interpreter):
        """Test list operations work in sandbox."""
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
    async def test_context_manager(self):
        """Test async context manager protocol."""
        interpreter = DenoInterpreter()

        async with interpreter:
            # Process should be running
            assert interpreter._deno_process is not None

        # Process should be stopped
        assert interpreter._deno_process is None

    @pytest.mark.asyncio
    async def test_sandbox_isolation(self, interpreter):
        """Test that Pyodide runs in an isolated virtual filesystem.

        Pyodide provides a virtual filesystem (/tmp, /home, /dev, /proc, /lib)
        that is separate from the host filesystem. The 'os' module works but
        only sees the virtual filesystem, not the actual host files.
        """
        async with interpreter:
            result = await interpreter.execute(
                code="import os; print(os.listdir('/'))",
                variables={},
                tools={},
            )

        # Pyodide's os module works but shows virtual filesystem
        assert result["success"] is True
        stdout = result["stdout"]
        # Should see Pyodide's virtual directories, not host directories
        assert "tmp" in stdout or "home" in stdout or "lib" in stdout
        # Should NOT see actual host directories like /Users, /Applications
        assert "Users" not in stdout
        assert "Applications" not in stdout

    @pytest.mark.asyncio
    async def test_dict_variable_injection(self, interpreter):
        """Test that dict variables are properly serialized."""
        async with interpreter:
            result = await interpreter.execute(
                code="print(data['key'])",
                variables={"data": {"key": "value", "num": 42}},
                tools={},
            )

        assert result["success"] is True
        assert "value" in result["stdout"]

    @pytest.mark.asyncio
    async def test_list_variable_injection(self, interpreter):
        """Test that list variables are properly serialized."""
        async with interpreter:
            result = await interpreter.execute(
                code="print(sum(numbers))",
                variables={"numbers": [1, 2, 3, 4, 5]},
                tools={},
            )

        assert result["success"] is True
        assert "15" in result["stdout"]

    @pytest.mark.asyncio
    async def test_none_variable(self, interpreter):
        """Test that None is properly serialized."""
        async with interpreter:
            result = await interpreter.execute(
                code="print(value is None)",
                variables={"value": None},
                tools={},
            )

        assert result["success"] is True
        assert "True" in result["stdout"]

    @pytest.mark.asyncio
    async def test_bool_variables(self, interpreter):
        """Test that bool variables are properly serialized."""
        async with interpreter:
            result = await interpreter.execute(
                code="print(flag1, flag2)",
                variables={"flag1": True, "flag2": False},
                tools={},
            )

        assert result["success"] is True
        assert "True False" in result["stdout"]

    @pytest.mark.asyncio
    async def test_multiple_executions(self, interpreter):
        """Test multiple sequential executions."""
        async with interpreter:
            result1 = await interpreter.execute(
                code="print('first')",
                variables={},
                tools={},
            )
            assert result1["success"] is True
            assert "first" in result1["stdout"]

            result2 = await interpreter.execute(
                code="print('second')",
                variables={},
                tools={},
            )
            assert result2["success"] is True
            assert "second" in result2["stdout"]

    @pytest.mark.asyncio
    async def test_get_namespace_empty(self, interpreter):
        """Test that get_namespace returns empty dict for Deno."""
        async with interpreter:
            await interpreter.execute(
                code="my_var = 123",
                variables={},
                tools={},
            )

            namespace = interpreter.get_namespace()
            # Deno sandbox doesn't support namespace introspection
            assert namespace == {}


class TestDenoInterpreterThreadSafety:
    """Thread-safety tests for DenoInterpreter."""

    @pytest.mark.asyncio
    async def test_thread_ownership_set_on_first_execute(self):
        """Test that thread ownership is set on first execute."""
        import threading

        interpreter = DenoInterpreter()
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
        interpreter = DenoInterpreter()

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

        interpreter = DenoInterpreter()

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
