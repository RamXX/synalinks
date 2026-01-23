# License Apache 2.0: (c) 2025 Synalinks Team

"""Reasoning modules for advanced LLM-based computation."""

from synalinks.src.modules.reasoning.repl_types import REPLEntry
from synalinks.src.modules.reasoning.repl_types import REPLHistory
from synalinks.src.modules.reasoning.repl_types import REPLVariable
from synalinks.src.modules.reasoning.repl_generator import REPLAction
from synalinks.src.modules.reasoning.repl_generator import REPLGenerator
from synalinks.src.modules.reasoning.repl_module import RLM
from synalinks.src.modules.reasoning.repl_module import REPLModule  # Backward compat

__all__ = [
    "REPLEntry",
    "REPLHistory",
    "REPLVariable",
    "REPLAction",
    "REPLGenerator",
    "RLM",
    "REPLModule",  # Backward-compatible alias for RLM
]
