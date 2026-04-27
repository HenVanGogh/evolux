## Summary

<!-- What does this PR do? Reference issue: Fixes #N -->

## Scope

- Module(s): <!-- e.g. brain, memory -->
- SPEC.md sections satisfied: <!-- e.g. transformer.py + acceptance test 1-3 -->

## Quality gates

- [ ] Only files in module scope edited (no cross-module changes)
- [ ] `ruff check .` clean
- [ ] `ruff format --check .` clean
- [ ] `python scripts/check_layering.py` passes
- [ ] `pytest tests/unit/<module>/` green
- [ ] No new dependencies, OR new deps declared in `pyproject.toml` and justified below
- [ ] Protocols not modified, OR Protocol-change procedure followed (see AGENTS.md §9)
- [ ] No `print` / silent `except` introduced

## New dependencies (if any)

<!-- name, version, why -->

## Notes for reviewers

<!-- Anything reviewers should pay extra attention to -->
