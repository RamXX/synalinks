# License Apache 2.0: (c) 2025 Synalinks Team

"""Abstract base class for code execution backends."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from synalinks.src.api_export import synalinks_export


class CodeInterpreterError(Exception):
    """Exception raised for interpreter errors."""

    pass


class FinalOutput:
    """Container for final output from SUBMIT() call.

    This is returned when the code calls SUBMIT() to signal completion.
    """

    def __init__(self, values: Dict[str, Any]):
        self.values = values

    def __repr__(self) -> str:
        return f"FinalOutput({self.values!r})"


@synalinks_export("synalinks.interpreters.CodeInterpreter")
class CodeInterpreter(ABC):
    """Abstract base for code execution backends.

    CodeInterpreter defines the interface for executing Python code
    in various environments (native, container, remote, etc.).

    Implementations must provide:
    - `execute()`: Run code with variables and tools
    - `start()`: Initialize the interpreter
    - `stop()`: Clean up resources

    Example:

    ```python
    async with interpreter:
        result = await interpreter.execute(
            code="print(x + y)",
            variables={"x": 1, "y": 2},
            tools={}
        )
        print(result["stdout"])  # "3"
    ```
    """

    @abstractmethod
    async def execute(
        self,
        code: str,
        variables: Dict[str, Any],
        tools: Dict[str, Callable],
    ) -> Dict[str, Any]:
        """Execute code and return result.

        Args:
            code: Python code to execute.
            variables: Variables to inject into namespace.
            tools: Callable tools to make available.

        Returns:
            Dict with keys:
                - success (bool): Whether execution succeeded
                - stdout (str): Captured stdout
                - stderr (str): Captured stderr
                - error (Optional[str]): Error message if any
                - submitted (Optional[dict]): Values from SUBMIT() call
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """Initialize the interpreter."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Clean up the interpreter."""
        pass

    async def __aenter__(self) -> "CodeInterpreter":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
