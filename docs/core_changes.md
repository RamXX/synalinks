# Core Changes Log

This log records changes made to Synalinks core modules. Each entry includes rationale,
implementation details, and expected impact to help maintainers review risk in
production deployments.

## 2026-01-23 — Spawn-safe tracking predicates and collections

### Rationale
On macOS, multiprocessing uses the spawn start method. During spawn unpickling,
Python can invoke list operations on subclasses (e.g., `list.extend`) before the
subclass `__init__` runs. The previous implementation stored a `tracker` attribute
only in `__init__` and used local `lambda` functions inside `Module._initialize_tracker`.
This combination caused:
- `AttributeError: 'TrackedList' object has no attribute 'tracker'` during spawn.
- Potential pickling issues because locally defined lambdas are not picklable.

These failures surfaced in `python_synthesis_test` when the REPL execution process
was spawned on macOS.

### Changes
- `synalinks/src/modules/module.py`
  - Moved tracking predicate lambdas to top-level functions (`_is_trainable_variable`,
    `_is_non_trainable_variable`, `_is_metric`, `_is_module`) to keep them picklable.
- `synalinks/src/utils/tracking.py`
  - Added class-level `tracker = None` defaults for `TrackedList`, `TrackedDict`,
    `TrackedSet`.
  - Guarded tracker access via `getattr(self, "tracker", None)` to avoid attribute
    errors if list operations execute before `__init__` during unpickling.

### Impact
- No functional behavior change for normal runtime tracking.
- Improves multiprocessing reliability on macOS (spawn).
- Reduces risk of pickling errors in core tracking utilities used by production
  systems.

### Verification
- `uv run pytest synalinks/src/modules/synthesis/python_synthesis_test.py -v` (2 passed)
- `uv run pytest --cov-config=pyproject.toml` (macOS: 557 passed, 4 skipped)

## 2026-01-24 — RLM required-field validation honors schema

### Rationale
RLM previously treated every output property as required when validating SUBMIT
results. This conflicted with Synalinks DataModels that mark fields as optional
via JSON Schema `required`. It also caused avoidable retries and failures when
optional outputs were omitted.

### Changes
- `synalinks/src/modules/reasoning/repl_module.py`
  - Added `required_fields` derived from schema `required` (fallback to all
    properties when absent).
  - Missing-field validation and fallback extraction prompts now reference
    `required_fields` instead of all properties.
- `tests/modules/reasoning/test_repl_module.py`
  - Added coverage for optional outputs being omitted without errors.

### Impact
- Optional output fields can be omitted without triggering validation errors.
- Required output fields continue to be enforced, preserving DSPy-style behavior
  when all outputs are required.

### Verification
- `uv run pytest tests/modules/reasoning/test_repl_module.py -q`
- `uv run --env-file .env -- python -m pytest tests/modules/reasoning/test_repl_module_integration.py -v --override-ini="addopts="`

## 2026-01-24 — Configurable strict JSON guidance for RLM instructions

### Rationale
Strict JSON/backslash rules are essential for some providers (e.g., Groq), but
they are overly constraining for others and diverge from DSPy’s default RLM
instructions. Making strict behavior configurable avoids unintended global
constraints while preserving Groq-safe operation when needed.

### Changes
- `synalinks/src/language_models/language_model.py`
  - Added `strict_json` option (defaults to True for Groq, False otherwise) and
    persisted it in serialization config.
  - Groq JSON guard injection now respects `strict_json`.
- `synalinks/src/modules/reasoning/repl_generator.py`
  - Instruction rules are now generated dynamically with strict JSON rules
    included only when `strict_json` is enabled.
  - code_lines mode selection now follows `strict_json`.
- `tests/modules/reasoning/test_repl_generator.py`
  - Added coverage for strict JSON rule inclusion/exclusion and code_lines mode.
- `tests/language_models/test_language_model.py`
  - Added coverage for strict_json defaults and overrides.

### Impact
- Non-Groq models receive DSPy-style RLM instructions by default.
- Groq-specific strict JSON guidance can be enabled explicitly or via default
  Groq inference to improve structured-output reliability.

### Verification
- `uv run pytest tests/modules/reasoning/test_repl_generator.py -q`
- `uv run pytest tests/language_models/test_language_model.py -q`
- `uv run --env-file .env -- python -m pytest tests/language_models/test_language_model_integration.py -v --override-ini="addopts="`

## 2026-01-24 — Safer line-by-line REPL fallback

### Rationale
RLM’s syntax stabilizer can execute code line-by-line after a SyntaxError, but
the previous guard only checked indentation/colons. This allowed partial
execution of multi-line constructs (open delimiters or backslash continuations),
which caused cascaded errors and noisy retries.

### Changes
- `synalinks/src/modules/reasoning/repl_module.py`
  - `_can_execute_line_by_line` now rejects backslash continuations, unbalanced
    delimiters, and newline tokens inside open delimiters (via tokenize), while
    preserving the original indentation/colon checks.
- `tests/modules/reasoning/test_repl_module.py`
  - Added coverage for multiline delimiters, backslash continuation, and simple
    single-line acceptance.

### Impact
- Line-by-line recovery only triggers for safe single-line snippets.
- Reduces false recovery attempts and cascading syntax failures.

### Verification
- `uv run pytest tests/modules/reasoning/test_repl_module.py -q`
- `uv run pytest tests/interpreters/test_native_interpreter.py -q`

## 2026-01-24 — Native interpreter BACKSLASH helper

### Rationale
Strict JSON REPL guidance instructs using a BACKSLASH helper to avoid literal
backslashes in JSON strings. The interpreter needs to preload this helper to
prevent NameError and keep guidance accurate.

### Changes
- `synalinks/src/interpreters/native.py`
  - Added `BACKSLASH = chr(92)` to the interpreter namespace at startup.
- `tests/interpreters/test_native_interpreter.py`
  - Added coverage for BACKSLASH availability.

### Impact
- REPL code can safely construct backslashes without embedding them in JSON.
- Aligns interpreter behavior with strict JSON guidance.

### Verification
- `uv run pytest tests/interpreters/test_native_interpreter.py -q`

## 2026-01-24 — RLM prompt includes environment limits and output types

### Rationale
Recent RLM runs wasted iterations on forbidden imports and type mismatches.
Making environment permissions explicit and adding output field type hints
reduces errors and aligns the prompt with DSPy-style typed signatures.

### Changes
- `synalinks/src/modules/reasoning/repl_generator.py`
  - Prompt rules now explicitly call out import/file I/O restrictions.
  - Output fields include required/optional markers and type hints derived
    from the output schema.
- `tests/modules/reasoning/test_repl_generator.py`
  - Added coverage for environment limits and type/requiredness formatting.

### Impact
- Fewer wasted iterations on forbidden imports.
- Reduced type errors during SUBMIT by giving explicit field types.

### Verification
- `uv run pytest tests/modules/reasoning/test_repl_generator.py -q`
- `uv run --env-file .env -- python -m pytest tests/modules/reasoning/test_repl_module_integration.py -v --override-ini="addopts="`

## 2026-01-24 — Groq failed_generation repair fallback in LanguageModel

### Rationale
Groq can return json_validate_failed with a failed_generation payload that
isn't valid JSON. Previously this collapsed to a retry stub, causing wasted
RLM iterations and higher cost. The LanguageModel now attempts a JSON repair
on the failed_generation text before giving up.

### Changes
- `synalinks/src/language_models/language_model.py`
  - Added a Groq-specific repair fallback when failed_generation cannot be
    parsed, reusing structured-output validation and DataModel checks.
- `tests/language_models/test_language_model.py`
  - Added coverage to ensure the repair path returns validated output.

### Impact
- Fewer Groq json_validate_failed loops when the payload is malformed.
- Improved resilience for strict JSON providers without changing non-Groq flows.

### Verification
- `uv run pytest tests/language_models/test_language_model.py -q`
- `uv run --env-file .env -- python -m pytest tests/language_models/test_language_model_integration.py -v --override-ini="addopts="`
