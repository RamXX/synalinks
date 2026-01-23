# REPLGenerator

The REPLGenerator is a specialized Generator that produces code and reasoning for each REPL iteration.

## Features

- Generates `REPLAction` outputs with `reasoning` and `code` fields
- Includes behavioral rules adapted from DSPy's RLM implementation
- Trainable instructions for optimization
- Code fence stripping for LLM output cleanup

## Attribution

Based on DSPy's RLM implementation. See [DSPy](https://github.com/stanfordnlp/dspy).

::: synalinks.src.modules.reasoning.repl_generator
