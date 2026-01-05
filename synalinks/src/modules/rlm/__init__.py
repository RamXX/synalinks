"""RLM (Recursive Language Model) integration for Synalinks.

This package provides RecursiveGenerator and related components for enabling
LLMs to execute Python code in sandboxed REPLs and make recursive sub-LM calls.
"""

from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator

__all__ = ["RecursiveGenerator"]
