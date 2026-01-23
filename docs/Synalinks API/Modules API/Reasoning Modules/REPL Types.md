# REPL Types

Data models for REPL execution history and state.

## REPLVariable

Rich metadata about variables available in the REPL environment. Used to provide context to the LLM without serializing full values into the prompt.

## REPLEntry

A single REPL iteration entry (immutable). Captures reasoning, code, and output from one iteration.

## REPLHistory

Complete REPL execution history (immutable). The `append()` method returns a new instance, preserving the original.

## Attribution

Based on DSPy's RLM implementation. See [DSPy](https://github.com/stanfordnlp/dspy).

::: synalinks.src.modules.reasoning.repl_types
