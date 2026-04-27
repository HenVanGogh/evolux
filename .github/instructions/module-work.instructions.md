---
applyTo: "src/evolux/**/*.py"
---

# Module-work instructions

When editing files under `src/evolux/<module>/`:

1. **First** read `src/evolux/<module>/SPEC.md` end-to-end.
2. Implement only what the issue or SPEC asks for. Don't expand scope.
3. Imports from other `evolux.*` modules are limited to **strictly lower layers** (see `docs/ARCHITECTURE.md`). Cross-module access goes through Protocols in `evolux.core.protocols`.
4. Add or update tests under `tests/unit/<module>/`. Use the `rng` fixture from `tests/conftest.py` for any randomness.
5. Run before committing:
   - `ruff check . --fix && ruff format .`
   - `python scripts/check_layering.py`
   - `pytest tests/unit/<module>/ -q`
6. If you find yourself wanting to change a Protocol in `evolux.core.protocols`, **stop** and follow AGENTS.md §9 instead.
