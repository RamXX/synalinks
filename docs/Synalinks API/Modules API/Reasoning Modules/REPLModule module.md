# RLM (Recursive Language Model)

The RLM class enables LLMs to solve complex problems by writing and executing Python code iteratively.

Note: `REPLModule` is available as a backward-compatible alias for `RLM`.

## Pass-by-Reference for Large Data

A key feature is that input variables are passed **by reference** into the REPL namespace, NOT serialized into the prompt. This enables processing of large data (entire books, large documents) without prompt bloat.

Only a small preview (default 500 characters) appears in the prompt to help the LLM understand data structure. The full data is accessible as a Python variable in the REPL environment.

**Example**: A 500KB document passed as `document` will:

- Show a 500-char preview in the prompt context
- Be fully accessible as `document` variable in code
- NOT consume 500KB of prompt tokens

## Attribution

Based on DSPy's RLM implementation. See [DSPy](https://github.com/stanfordnlp/dspy).

::: synalinks.src.modules.reasoning.repl_module
