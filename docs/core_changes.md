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

## 2026-01-24 — LanguageModel strict_json option

### Rationale
Strict JSON/backslash rules are essential for some providers (e.g., Groq), but
they are overly constraining for others. Making strict behavior configurable
avoids unintended global constraints while preserving Groq-safe operation when
needed.

### Changes
- `synalinks/src/language_models/language_model.py`
  - Added `strict_json` option (defaults to True for Groq, False otherwise) and
    persisted it in serialization config.
  - Groq JSON guard injection now respects `strict_json`.
- `tests/language_models/test_language_model.py`
  - Added coverage for strict_json defaults and overrides.

### Impact
- Non-Groq models are not subject to strict JSON constraints by default.
- Groq-specific strict JSON guidance can be enabled explicitly or via default
  Groq inference to improve structured-output reliability.

### Verification
- `uv run pytest tests/language_models/test_language_model.py -q`
- `uv run --env-file .env -- python -m pytest tests/language_models/test_language_model_integration.py -v --override-ini="addopts="`

## 2026-01-24 — Groq failed_generation repair fallback in LanguageModel

### Rationale
Groq can return json_validate_failed with a failed_generation payload that
isn't valid JSON. Previously this collapsed to a retry stub, causing wasted
iterations and higher cost. The LanguageModel now attempts a JSON repair
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
