"""Local REPL for executing Python code in sandboxed environment."""

import asyncio
import io
import json
import os
import shutil
import signal
import tempfile
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from typing import Any
from typing import Callable
from typing import Optional

from synalinks.src.modules.rlm.core.types import REPLResult


class LocalREPL:
    """Executes Python code in sandboxed environment with safe builtins.

    Provides a minimal execution environment with standard builtins but no
    file system access, network access, or other dangerous operations.

    Supports multi-model architecture via default_sub_model parameter for
    cost-optimized llm_query() routing to cheaper sub-models.

    Tracks recursion depth for nested llm_query calls and enforces max_depth limit.

    Args:
        llm_query_fn: Optional function to inject as llm_query() builtin
        default_sub_model: Default client name for llm_query() routing
        current_depth: Current recursion depth (default: 0)
        max_depth: Maximum allowed recursion depth (default: 1)

    Example:
        >>> # Basic usage
        >>> repl = LocalREPL()
        >>> result = repl.execute("x = 1 + 2\\nprint(x)")
        >>> result.stdout
        '3\\n'
        >>> result.locals['x']
        3

        >>> # Multi-model usage with default routing to sub-model
        >>> handler = LMHandler()
        >>> handler.register_client("root", client_root)
        >>> handler.register_client("sub", client_sub)
        >>> llm_query_fn = handler.create_llm_query_fn("sub")  # Route to sub
        >>> repl = LocalREPL(llm_query_fn=llm_query_fn, default_sub_model="sub")
        >>> result = repl.execute("answer = llm_query('What is 2+2?')")

        >>> # Recursion depth tracking
        >>> repl = LocalREPL(llm_query_fn=llm_query_fn, current_depth=0, max_depth=2)
        >>> # llm_query calls can nest up to depth 2
    """

    def __init__(
        self,
        llm_query_fn: Optional[Callable] = None,
        default_sub_model: Optional[str] = None,
        llm_query_batched_fn: Optional[Callable] = None,
        timeout: Optional[float] = None,
        current_depth: int = 0,
        max_depth: int = 1,
    ):
        """Initialize REPL with safe builtins.

        Args:
            llm_query_fn: Function to inject as llm_query() for LM calls
            default_sub_model: Default client name for llm_query() routing
            llm_query_batched_fn: Function to inject as llm_query_batched()
                for batched calls
            timeout: Optional timeout in seconds for code execution
            current_depth: Current recursion depth (default: 0)
            max_depth: Maximum allowed recursion depth (default: 1)
        """
        self._locals: dict[str, Any] = {}
        self.default_sub_model = default_sub_model
        self.timeout = timeout
        self.current_depth = current_depth
        self.max_depth = max_depth
        self._final_answer: Optional[Any] = None
        self.temp_dir = tempfile.mkdtemp()
        self._init_builtins(llm_query_fn, llm_query_batched_fn)

    def _create_final_var_fn(self):
        """Create FINAL_VAR function that captures structured results.

        Returns:
            Function that sets the final answer value
        """

        def FINAL_VAR(value: Any) -> Any:
            """Set the final answer to be returned in REPLResult.

            Args:
                value: The structured result to return

            Returns:
                The value (for convenience in expressions)
            """
            self._final_answer = value
            return value

        return FINAL_VAR

    def _create_llm_query_batched_wrapper(
        self, llm_query_fn: Optional[Callable]
    ) -> Optional[Callable]:
        """Create async wrapper for batched LLM queries using asyncio.gather.

        Args:
            llm_query_fn: The base llm_query function to wrap

        Returns:
            Batched query function or None if llm_query_fn is None
        """
        if llm_query_fn is None:
            return None

        def llm_query_batched(prompts: list[str]) -> list[Any]:
            """Execute multiple LLM queries concurrently using asyncio.gather.

            Args:
                prompts: List of prompts to query

            Returns:
                List of responses in same order as prompts
            """

            async def _query_all():
                tasks = [asyncio.to_thread(llm_query_fn, p) for p in prompts]
                return await asyncio.gather(*tasks)

            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(_query_all())

        return llm_query_batched

    def _init_builtins(
        self,
        llm_query_fn: Optional[Callable],
        llm_query_batched_fn: Optional[Callable] = None,
    ):
        """Initialize safe builtins environment.

        Provides standard Python builtins but restricts dangerous operations.
        """
        # Safe builtins - exclude open, eval, exec, etc.
        # Note: __import__ is allowed for importing safe modules like time, math
        safe_builtins = {
            "__import__": __import__,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
        }

        # Add FINAL_VAR function
        safe_builtins["FINAL_VAR"] = self._create_final_var_fn()

        # Add llm_query if provided
        if llm_query_fn:
            safe_builtins["llm_query"] = llm_query_fn

        # Add llm_query_batched - use provided or create wrapper
        if llm_query_batched_fn:
            safe_builtins["llm_query_batched"] = llm_query_batched_fn
        elif llm_query_fn:
            # Auto-create batched version using asyncio.gather
            safe_builtins["llm_query_batched"] = self._create_llm_query_batched_wrapper(
                llm_query_fn
            )

        self._locals["__builtins__"] = safe_builtins

    def execute(self, code: str, timeout: Optional[float] = None) -> REPLResult:
        """Execute Python code and return result.

        Args:
            code: Python code to execute
            timeout: Optional timeout in seconds (overrides instance timeout)

        Returns:
            REPLResult containing stdout, stderr, locals, exception, final_answer

        Example:
            >>> repl = LocalREPL()
            >>> result = repl.execute("x = 42")
            >>> result.success
            True
            >>> result.locals['x']
            42
        """
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exception = None
        self._final_answer = None  # Reset final answer

        # Use provided timeout or instance timeout
        exec_timeout = timeout if timeout is not None else self.timeout

        def _execute():
            """Inner execution function."""
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self._locals)

        try:
            if exec_timeout is not None:
                # Set up timeout using signal (Unix-only)
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Execution exceeded {exec_timeout}s timeout")

                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, exec_timeout)
                try:
                    _execute()
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                _execute()
        except Exception as e:
            exception = e
            stderr_buffer.write(f"{type(e).__name__}: {e}\n")

        return REPLResult(
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            locals=self._locals.copy(),
            exception=exception,
            final_answer=self._final_answer,
        )

    def get_variable(self, name: str) -> Any:
        """Get value of variable from REPL state.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable doesn't exist
        """
        if name not in self._locals:
            raise KeyError(f"Variable '{name}' not found in REPL")
        return self._locals[name]

    def reset(
        self,
        llm_query_fn: Optional[Callable] = None,
        default_sub_model: Optional[str] = None,
        llm_query_batched_fn: Optional[Callable] = None,
        timeout: Optional[float] = None,
        current_depth: Optional[int] = None,
        max_depth: Optional[int] = None,
    ):
        """Reset REPL state to fresh environment.

        Args:
            llm_query_fn: Optional new llm_query function
            default_sub_model: Optional new default client name for routing
            llm_query_batched_fn: Optional new llm_query_batched function
            timeout: Optional new timeout in seconds
            current_depth: Optional new current recursion depth
            max_depth: Optional new maximum recursion depth
        """
        self._locals.clear()
        self._final_answer = None
        if default_sub_model is not None:
            self.default_sub_model = default_sub_model
        if timeout is not None:
            self.timeout = timeout
        if current_depth is not None:
            self.current_depth = current_depth
        if max_depth is not None:
            self.max_depth = max_depth
        self._init_builtins(llm_query_fn, llm_query_batched_fn)

    def load_context(self, context: str | dict) -> None:
        """Load context into the REPL namespace.

        Args:
            context: Either a JSON string or dict to load into namespace

        Example:
            >>> repl = LocalREPL()
            >>> repl.load_context({"x": 42, "y": "hello"})
            >>> repl.get_variable("x")
            42
            >>> repl.load_context('{"z": [1, 2, 3]}')
            >>> repl.get_variable("z")
            [1, 2, 3]
        """
        if isinstance(context, str):
            context = json.loads(context)
        self._locals.update(context)

    def cleanup(self) -> None:
        """Clean up temp directory."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def __del__(self):
        """Clean up resources on deletion."""
        self.cleanup()
