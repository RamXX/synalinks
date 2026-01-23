# Reasoning Modules

Reasoning modules enable LLMs to solve complex problems through iterative code execution and reasoning. These modules are based on the Reasoning Language Models (RLM) implementation from [DSPy](https://github.com/stanfordnlp/dspy).

## Overview

The RLM (Recursive Language Model) class implements a Read-Eval-Print Loop where:

1. The LLM writes Python code to explore data and solve problems
2. Code is executed in a sandboxed interpreter
3. The LLM observes outputs and writes more code
4. This continues until the LLM calls `SUBMIT()` with the final answer

## Key Features

- **Pass-by-Reference**: Large inputs (documents, books) are injected into the REPL namespace by reference, not serialized into prompts
- **Sub-LLM Queries**: `llm_query()` enables semantic analysis of data portions
- **Custom Tools**: Domain-specific tools can be injected
- **Trainable**: Instructions can be optimized via Synalinks training

## Modules

- [RLM](REPLModule module.md) - Main reasoning module with REPL execution loop (REPLModule is a backward-compatible alias)
- [REPLGenerator](REPLGenerator module.md) - Code generation component
- [REPL Types](REPL Types.md) - Data types for REPL state (REPLVariable, REPLEntry, REPLHistory)

## Attribution

These modules are based on DSPy's RLM implementation. The techniques and patterns were reimplemented to integrate with Synalinks' architecture while preserving the core RLM concepts.

Reference: [DSPy - Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://github.com/stanfordnlp/dspy)
