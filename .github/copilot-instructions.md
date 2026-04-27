# Repo-wide instructions for AI coding agents

You are contributing to **evolux**, a GPU-native modular framework for the simulated evolution of intelligent creatures.

## Before doing anything

1. Read the module's `SPEC.md` (e.g. `src/evolux/brain/SPEC.md`).
2. Read [AGENTS.md](../AGENTS.md) for the contract and hard rules.
3. Stay inside the module's "Files in scope". Do **not** edit other modules unless the issue explicitly says so.

## Hard rules (CI-enforced)

- Cross-module communication is **only via Protocols** in `evolux.core.protocols`. Never import a concrete class from another module.
- Layering: a module may only import modules at strictly lower layers (see `docs/ARCHITECTURE.md`). Enforced by `scripts/check_layering.py`.
- All public functions/classes have type hints.
- No `print`. Use `rich` or `logging`.
- No silent `except:` / `except Exception: pass`.
- Randomness goes through `evolux.core.rng.RNG`. Do not call `torch.manual_seed` directly in module code.
- Tests pass on CPU. GPU-only tests must be marked `@pytest.mark.gpu`.

## Style

- Format & lint: `ruff format` + `ruff check`. Line length 100.
- Conventional commits: `feat(brain): ...`, `fix(memory): ...`, `test(evolution): ...`.

## When in doubt

- Don't expand scope.
- Open a question in the PR description rather than guessing about other modules' APIs.
- For Protocol changes, follow the procedure in AGENTS.md §9 (rename + alias + CHANGELOG entry).
