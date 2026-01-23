# License Apache 2.0: (c) 2025 Synalinks Team

"""Native Python interpreter with tool injection.

This interpreter is based on the Reasoning Language Models (RLM) implementation
from DSPy (https://github.com/stanfordnlp/dspy). The SUBMIT mechanism, builtin
restrictions, and thread-safety patterns were adapted from DSPy's implementation.

Reference:
    DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
    https://github.com/stanfordnlp/dspy
"""

import asyncio
import builtins
import collections
import inspect
import io
import json
import math
import re
import threading
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.interpreters.base import CodeInterpreter


class SubmitCalled(Exception):
    """Exception raised when SUBMIT() is called to signal completion."""

    def __init__(self, values: Dict[str, Any]):
        self.values = values
        super().__init__("SUBMIT called")


@synalinks_export("synalinks.interpreters.NativePythonInterpreter")
class NativePythonInterpreter(CodeInterpreter):
    """Native Python interpreter with tool injection.

    Tools are executed in the host environment while user code
    runs in a restricted namespace with limited builtins.

    Example:

    ```python
    interpreter = NativePythonInterpreter()

    async with interpreter:
        result = await interpreter.execute(
            code="result = x * 2\\nprint(result)",
            variables={"x": 21},
            tools={}
        )
        print(result["stdout"])  # "42"
    ```

    Args:
        allowed_builtins: List of builtin names to allow.
            Defaults to a safe subset.
        max_output_chars: Maximum characters to capture from stdout.
            Defaults to 100,000.
    """

    DEFAULT_BUILTINS = [
        # Type constructors
        "list",
        "dict",
        "set",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "frozenset",
        "complex",
        # Iteration
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "iter",
        "next",
        "reversed",
        "sorted",
        "slice",
        # Aggregation
        "sum",
        "min",
        "max",
        "abs",
        "round",
        "pow",
        "divmod",
        # Boolean
        "any",
        "all",
        "bool",
        # Type checking
        "isinstance",
        "issubclass",
        "type",
        "callable",
        # Attribute access
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        # Object
        "id",
        "hash",
        "repr",
        "ascii",
        "chr",
        "ord",
        "hex",
        "oct",
        "bin",
        "format",
        # I/O
        "print",
        "input",
        # Other
        "vars",
        "dir",
        "locals",
        "globals",
        "staticmethod",
        "classmethod",
        "property",
        "super",
        "object",
        # Exceptions (needed for error handling in user code)
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "StopIteration",
        "AssertionError",
        "ZeroDivisionError",
        "NotImplementedError",
    ]
    DEFAULT_MODULES = {
        "collections": collections,
        "json": json,
        "math": math,
        "re": re,
    }
    ALLOWED_IMPORTS = set(DEFAULT_MODULES.keys())

    def __init__(
        self,
        allowed_builtins: Optional[List[str]] = None,
        max_output_chars: int = 100_000,
    ):
        self.allowed_builtins = allowed_builtins or self.DEFAULT_BUILTINS
        self.max_output_chars = max_output_chars
        self._namespace: Dict[str, Any] = {}
        self._started = False
        self._owner_thread: Optional[int] = None

    def _check_thread_ownership(self) -> None:
        """Ensure this interpreter is only used from a single thread.

        NativePythonInterpreter is not thread-safe and cannot be shared
        across threads. This method enforces single-thread ownership.

        Raises:
            RuntimeError: If called from a different thread than the owner.
        """
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "NativePythonInterpreter is not thread-safe and cannot be shared "
                "across threads. Create a separate interpreter instance for each thread."
            )

    async def start(self) -> None:
        """Initialize the interpreter with a fresh namespace."""
        # Preload safe stdlib modules for use without imports.
        self._namespace = dict(self.DEFAULT_MODULES)
        self._started = True

    async def stop(self) -> None:
        """Clear the namespace."""
        self._namespace.clear()
        self._started = False
        self._owner_thread = None

    def _build_builtins(self) -> Dict[str, Any]:
        """Build restricted builtins dict."""
        restricted = {}
        for name in self.allowed_builtins:
            if hasattr(builtins, name):
                restricted[name] = getattr(builtins, name)
        # Provide a restricted import mechanism for safe modules only.
        restricted["__import__"] = self._safe_import
        return restricted

    def _safe_import(
        self,
        name: str,
        globals: Optional[Dict[str, Any]] = None,
        locals: Optional[Dict[str, Any]] = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Allow importing only whitelisted stdlib modules."""
        if level != 0:
            raise ImportError("Relative imports are not allowed")
        if name in self.ALLOWED_IMPORTS:
            return self.DEFAULT_MODULES[name]
        raise ImportError(f"Import of '{name}' is not allowed")

    def _create_submit_function(self) -> tuple[Callable, Dict[str, Any]]:
        """Create SUBMIT function and container for submitted values."""
        submitted_values: Dict[str, Any] = {}

        def SUBMIT(**kwargs: Any) -> None:
            """Submit final output values.

            Call this function when you have computed the final answer.
            Pass all required output fields as keyword arguments.

            Example:
                SUBMIT(answer="The result is 42", confidence=0.95)
            """
            submitted_values.update(kwargs)
            raise SubmitCalled(kwargs)

        return SUBMIT, submitted_values

    def _wrap_tools_for_sync_exec(
        self, tools: Dict[str, Callable]
    ) -> Dict[str, Callable]:
        """Wrap async tools to be sync-callable via asyncio.run().

        Since exec() runs synchronously and can't await coroutines directly,
        we wrap async tools to run in a new event loop in a thread.
        """
        import concurrent.futures

        wrapped = {}
        for name, fn in tools.items():
            if asyncio.iscoroutinefunction(fn):
                # Wrap async function to be sync-callable
                def make_sync_wrapper(async_fn: Callable) -> Callable:
                    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                        coro = async_fn(*args, **kwargs)
                        # Run in a new thread with its own event loop
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(asyncio.run, coro)
                            return future.result()

                    # Preserve function metadata
                    sync_wrapper.__name__ = getattr(async_fn, "__name__", name)
                    sync_wrapper.__doc__ = getattr(async_fn, "__doc__", None)
                    return sync_wrapper

                wrapped[name] = make_sync_wrapper(fn)
            else:
                wrapped[name] = fn

        return wrapped

    async def execute(
        self,
        code: str,
        variables: Dict[str, Any],
        tools: Dict[str, Callable],
    ) -> Dict[str, Any]:
        """Execute code with injected variables and tools.

        Args:
            code: Python code to execute.
            variables: Variables to inject into namespace.
            tools: Callable tools to make available.

        Returns:
            Execution result dict.
        """
        self._check_thread_ownership()
        if not self._started:
            await self.start()

        # Create SUBMIT function
        submit_fn, submitted_values = self._create_submit_function()

        # Check if code uses explicit await - if so, run in async mode
        # with original tools; otherwise wrap async tools for sync execution
        is_async_code = "await " in code

        if is_async_code:
            # Keep tools as-is for async execution
            exec_tools = tools
        else:
            # Wrap async tools to be sync-callable
            exec_tools = self._wrap_tools_for_sync_exec(tools)

        # Build execution namespace
        namespace = {
            **self._namespace,
            **variables,
            **exec_tools,
            "SUBMIT": submit_fn,
            "__builtins__": self._build_builtins(),
        }

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                if is_async_code:
                    # Wrap in async function and execute
                    indented_code = "\n".join(
                        f"    {line}" if line.strip() else line
                        for line in code.split("\n")
                    )
                    async_wrapper = f"async def __async_exec():\n{indented_code}"
                    exec(async_wrapper, namespace)
                    await namespace["__async_exec"]()
                else:
                    exec(code, namespace)

            # Update persistent namespace with new variables
            # (excluding private, input variables, and tools)
            for key, value in namespace.items():
                if (
                    not key.startswith("_")
                    and key not in variables
                    and key not in exec_tools
                    and key != "SUBMIT"
                    and key != "__builtins__"
                ):
                    self._namespace[key] = value

            stdout = stdout_capture.getvalue()
            if len(stdout) > self.max_output_chars:
                stdout = (
                    stdout[: self.max_output_chars]
                    + f"\n... (truncated, {len(stdout)} total chars)"
                )

            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr_capture.getvalue(),
                "error": None,
                "submitted": None,
            }

        except SubmitCalled:
            stdout = stdout_capture.getvalue()
            if len(stdout) > self.max_output_chars:
                stdout = stdout[: self.max_output_chars]

            return {
                "success": True,
                "stdout": stdout,
                "stderr": "",
                "error": None,
                "submitted": submitted_values,
            }

        except Exception as e:
            stdout = stdout_capture.getvalue()
            if len(stdout) > self.max_output_chars:
                stdout = stdout[: self.max_output_chars]

            return {
                "success": False,
                "stdout": stdout,
                "stderr": stderr_capture.getvalue(),
                "error": f"{type(e).__name__}: {str(e)}",
                "submitted": None,
            }

    def get_namespace(self) -> Dict[str, Any]:
        """Get current persistent namespace (for debugging)."""
        return dict(self._namespace)

    def clear_namespace(self) -> None:
        """Clear the persistent namespace."""
        self._namespace.clear()
