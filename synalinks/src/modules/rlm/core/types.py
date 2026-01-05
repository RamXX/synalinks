"""Core data types for RLM."""

import json
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
        final_answer: Optional structured result set via FINAL_VAR()
    """

    stdout: str = ""
    stderr: str = ""
    locals: dict[str, Any] = field(default_factory=dict)
    exception: Optional[Exception] = None
    final_answer: Optional[Any] = None

    @property
    def success(self) -> bool:
        """Whether execution completed without exception."""
        return self.exception is None


@dataclass
class RLMSubCall:
    """Record of a sub-LM call during RLM execution.

    Attributes:
        model: Model identifier used for this call
        prompt: Input prompt sent to the model
        response: Response received from the model
        depth: Recursion depth at which this call occurred
        error: Optional error message if call failed
    """

    model: str
    prompt: str
    response: str
    depth: int
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the sub-call
        """
        return {
            "model": self.model,
            "prompt": self.prompt,
            "response": self.response,
            "depth": self.depth,
            "error": self.error,
        }


@dataclass
class RLMIteration:
    """Record of a single RLM loop iteration.

    Attributes:
        iteration: Iteration number (0-indexed)
        prompt: Full prompt sent to the model
        response: Raw response from the model
        code_blocks: List of code blocks extracted from response
        execution_results: List of REPLResult objects from code execution
        sub_calls: List of sub-LM calls made during this iteration
        final_answer: Optional final answer if iteration produced one
    """

    iteration: int
    prompt: str
    response: str
    code_blocks: list[str] = field(default_factory=list)
    execution_results: list[dict[str, Any]] = field(default_factory=list)
    sub_calls: list[RLMSubCall] = field(default_factory=list)
    final_answer: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the iteration
        """
        return {
            "iteration": self.iteration,
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": self.code_blocks,
            "execution_results": self.execution_results,
            "sub_calls": [sc.to_dict() for sc in self.sub_calls],
            "final_answer": self.final_answer,
        }


@dataclass
class RLMTrajectory:
    """Complete execution trajectory for an RLM run.

    Attributes:
        iterations: List of all iterations executed
        total_iterations: Total number of iterations executed
        root_model: Model used for root/orchestration calls
        sub_model: Model used for recursive sub-calls
        max_iterations: Maximum iterations allowed
        max_depth: Maximum recursion depth allowed
        success: Whether execution completed successfully
        error: Optional error message if execution failed
    """

    iterations: list[RLMIteration] = field(default_factory=list)
    total_iterations: int = 0
    root_model: str = ""
    sub_model: str = ""
    max_iterations: int = 30
    max_depth: int = 1
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the trajectory
        """
        return {
            "iterations": [it.to_dict() for it in self.iterations],
            "total_iterations": self.total_iterations,
            "root_model": self.root_model,
            "sub_model": self.sub_model,
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth,
            "success": self.success,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export trajectory as formatted JSON string.

        Args:
            indent: Number of spaces for JSON indentation (default: 2)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Export trajectory as human-readable markdown.

        Returns:
            Markdown-formatted string representation
        """
        lines = []
        lines.append("# RLM Execution Trajectory")
        lines.append("")
        lines.append(f"**Root Model**: {self.root_model}")
        lines.append(f"**Sub Model**: {self.sub_model}")
        lines.append(
            f"**Total Iterations**: {self.total_iterations}/{self.max_iterations}"
        )
        lines.append(f"**Max Depth**: {self.max_depth}")
        lines.append(f"**Success**: {self.success}")
        if self.error:
            lines.append(f"**Error**: {self.error}")
        lines.append("")

        for it in self.iterations:
            lines.append(f"## Iteration {it.iteration}")
            lines.append("")
            lines.append("### Prompt")
            lines.append("```")
            lines.append(it.prompt)
            lines.append("```")
            lines.append("")
            lines.append("### Response")
            lines.append("```")
            lines.append(it.response)
            lines.append("```")
            lines.append("")

            if it.code_blocks:
                lines.append("### Code Blocks")
                for i, code in enumerate(it.code_blocks):
                    lines.append(f"#### Block {i + 1}")
                    lines.append("```python")
                    lines.append(code)
                    lines.append("```")
                    lines.append("")

            if it.execution_results:
                lines.append("### Execution Results")
                for i, result in enumerate(it.execution_results):
                    lines.append(f"#### Result {i + 1}")
                    if result.get("stdout"):
                        lines.append("**stdout:**")
                        lines.append("```")
                        lines.append(result["stdout"])
                        lines.append("```")
                    if result.get("stderr"):
                        lines.append("**stderr:**")
                        lines.append("```")
                        lines.append(result["stderr"])
                        lines.append("```")
                    if result.get("exception"):
                        lines.append(f"**exception**: {result['exception']}")
                    lines.append("")

            if it.sub_calls:
                lines.append("### Sub-LM Calls")
                for i, sc in enumerate(it.sub_calls):
                    lines.append(
                        f"#### Call {i + 1} (depth={sc.depth}, model={sc.model})"
                    )
                    lines.append(f"**Prompt**: {sc.prompt[:100]}...")
                    lines.append(f"**Response**: {sc.response[:100]}...")
                    if sc.error:
                        lines.append(f"**Error**: {sc.error}")
                    lines.append("")

            if it.final_answer is not None:
                lines.append("### Final Answer")
                lines.append("```json")
                lines.append(json.dumps(it.final_answer, indent=2))
                lines.append("```")
                lines.append("")

        return "\n".join(lines)


@dataclass
class RLMExecutionMetrics:
    """Metrics from a single RecursiveGenerator execution.

    Tracks comprehensive usage statistics for cost analysis and optimization.
    Enables monitoring of multi-model architectures where root and sub-LM
    have different costs.

    Attributes:
        iteration_count: Number of RLM loop iterations executed
        sub_call_count: Number of recursive llm_query() calls made
        root_model_usage: Token usage for root/orchestration model
        sub_model_usage: Token usage for sub/recursive model
        total_tokens: Total tokens across all models
        prompt_tokens: Total prompt tokens across all models
        completion_tokens: Total completion tokens across all models

    Example:
        >>> metrics = generator.get_last_metrics()
        >>> print(f"Iterations: {metrics.iteration_count}")
        >>> print(f"Sub-calls: {metrics.sub_call_count}")
        >>> print(f"Total tokens: {metrics.total_tokens}")
        >>> print(f"Estimated cost: ${metrics.estimated_cost:.4f}")
    """

    iteration_count: int = 0
    sub_call_count: int = 0
    root_model_usage: dict[str, Any] = field(default_factory=dict)
    sub_model_usage: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimate execution cost in USD based on model pricing.

        Uses approximate pricing for common models:
        - zai/glm-4.7: $0.0001/1K tokens (input/output)
        - groq/openai/gpt-oss-20b: $0.0001/1K tokens (approximate)

        Returns:
            Estimated cost in USD

        Note:
            This is a rough estimate. Actual costs may vary by provider
            and pricing tier. Use provider billing for exact costs.
        """
        # Approximate pricing per 1K tokens (simplified)
        # In production, this should query actual provider pricing
        cost_per_1k_tokens = 0.0001

        # Calculate cost
        cost = (self.total_tokens / 1000.0) * cost_per_1k_tokens
        return cost
