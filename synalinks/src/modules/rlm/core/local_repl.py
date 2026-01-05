"""Local REPL for executing Python code in sandboxed environment."""

import io
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

    Args:
        llm_query_fn: Optional function to inject as llm_query() builtin

    Example:
        >>> repl = LocalREPL()
        >>> result = repl.execute("x = 1 + 2\\nprint(x)")
        >>> result.stdout
        '3\\n'
        >>> result.locals['x']
        3
    """

    def __init__(self, llm_query_fn: Optional[Callable] = None):
        """Initialize REPL with safe builtins.

        Args:
            llm_query_fn: Function to inject as llm_query() for LM calls
        """
        self._locals: dict[str, Any] = {}
        self._init_builtins(llm_query_fn)

    def _init_builtins(self, llm_query_fn: Optional[Callable]):
        """Initialize safe builtins environment.

        Provides standard Python builtins but restricts dangerous operations.
        """
        # Safe builtins - exclude open, eval, exec, __import__, etc.
        safe_builtins = {
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

        # Add llm_query if provided
        if llm_query_fn:
            safe_builtins["llm_query"] = llm_query_fn

        self._locals["__builtins__"] = safe_builtins

    def execute(self, code: str) -> REPLResult:
        """Execute Python code and return result.

        Args:
            code: Python code to execute

        Returns:
            REPLResult containing stdout, stderr, locals, and exception

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

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self._locals)
        except Exception as e:
            exception = e
            stderr_buffer.write(f"{type(e).__name__}: {e}\n")

        return REPLResult(
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            locals=self._locals.copy(),
            exception=exception,
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

    def reset(self, llm_query_fn: Optional[Callable] = None):
        """Reset REPL state to fresh environment.

        Args:
            llm_query_fn: Optional new llm_query function
        """
        self._locals.clear()
        self._init_builtins(llm_query_fn)
