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
