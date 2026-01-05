"""Core data types for RLM."""

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional


@dataclass
class REPLResult:
    """Result from REPL code execution.

    Attributes:
        stdout: Standard output from execution
        stderr: Standard error from execution
        locals: Local variables after execution
        exception: Exception object if execution failed, None otherwise
    """

    stdout: str = ""
    stderr: str = ""
    locals: dict[str, Any] = field(default_factory=dict)
    exception: Optional[Exception] = None

    @property
    def success(self) -> bool:
        """Whether execution completed without exception."""
        return self.exception is None
