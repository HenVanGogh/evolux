# Contributing to evolux

Whether you're a human contributor or an AI coding agent, the workflow is the same: see **[AGENTS.md](AGENTS.md)** for the full contract.

## TL;DR

1. Pick an issue tagged `module:<name>` from the [issues page](https://github.com/HenVanGogh/evolux/issues).
2. Read `docs/ARCHITECTURE.md`, `docs/INTERFACES.md`, and your module's `SPEC.md`.
3. Branch: `feat/<module>-<issue-number>`.
4. Implement, test, lint:
   ```bash
   ruff check . && ruff format --check .
   pytest tests/unit/<module>/ -q
   python scripts/check_layering.py
   ```
5. Open PR. Reference the issue with `Closes #N`.

## Style

- `ruff` is the source of truth (config in `ruff.toml`).
- Type hints required on public APIs.
- `logging`, never `print`.
- Tensor shape comments encouraged: `# (B, T, D)`.

## Tests

- Unit tests live in `tests/unit/<module>/`.
- Integration tests live in `tests/integration/`.
- Performance benchmarks in `tests/benchmarks/` (skipped by default; run with `pytest --benchmark`).

## Commit messages

Conventional Commits style:

```
feat(brain): add Mamba SSM block
fix(world): correct food respawn rate scaling
docs(roadmap): update Phase 2 status
test(memory): add episodic write/read invariants
chore(ci): bump action versions
```

## License

By contributing, you agree your contributions are licensed under MIT (see `LICENSE`).
