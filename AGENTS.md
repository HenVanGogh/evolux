# AGENTS.md — Cloud Agent Orchestration Guide

> **Audience:** automated coding agents (Copilot Coding Agent, Claude Code, Cursor background agents, etc.) working on `evolux` in parallel.
>
> **Read this file first.** Then read the SPEC.md of the module you've been assigned.

---

## 1. The contract

Every module in `src/evolux/<module>/` ships with a `SPEC.md` that defines:

- **Mission** — one paragraph
- **Protocol(s) implemented** — the typed contract
- **Files in scope** — what you may create/edit
- **Files out of scope** — what you must not touch
- **Acceptance tests** — what `pytest` calls must pass
- **Performance budget** — latency / VRAM targets
- **Dependencies allowed** — modules and pip packages you may import

If your code passes the acceptance tests AND respects every "out of scope" boundary AND meets the perf budget, your work is **done**. Open a PR linking to your assigned issue.

---

## 2. Workflow

```
1.  pick an unassigned GitHub issue tagged `module:<name>`
2.  assign yourself, move issue to "In progress"
3.  create branch:   feat/<module>-<issue-number>
4.  read:
      - docs/ARCHITECTURE.md   (always)
      - docs/INTERFACES.md     (always)
      - src/evolux/<your-module>/SPEC.md
      - src/evolux/<dep-module>/SPEC.md  for each `depends_on` entry
5.  implement against the Protocols, not concrete classes
6.  add unit tests under tests/unit/<module>/
7.  run:
      ruff check . && ruff format --check .
      pytest tests/unit/<module>/ -q
      python scripts/check_layering.py
8.  open PR. CI must be green. Reference the issue with "Closes #N".
```

---

## 3. Hard rules

These are CI-enforced or PR-blocked.

| Rule | Enforced by |
| --- | --- |
| No upward dependencies (see `docs/ARCHITECTURE.md` layering) | `scripts/check_layering.py` in CI |
| No edits outside your `files in scope` | PR review (label: `scope-violation`) |
| Public functions/classes must be type-hinted | `ruff` with `ANN` rules |
| Any new public symbol that crosses module boundary must be in `core.protocols` | PR review |
| Tests must pass on CPU (small batch) AND, if available, GPU | CI matrix |
| No `print()` in shipped code (use `logging`) | `ruff` |
| No silent `except Exception` | `ruff` |
| Determinism: any randomness must use `core.rng.RNG`, never `random` / unseeded `torch.rand*` | review |

---

## 4. Soft rules

- Prefer **functional style** for tensor ops; classes only for stateful components.
- Document non-obvious tensor shapes inline: `# (B, T, D)`.
- Add a docstring `Notes` section linking to relevant papers when implementing a known algorithm.
- One module per PR. If you must touch a sibling module's interface, open a `protocols/` PR first.

---

## 5. Module ownership board

Live status lives in [docs/MODULES.md](docs/MODULES.md). Update the **Owner agent** column when you start a module and **Status** as you progress through `scaffold → in-progress → tested → reviewed → merged`.

If two agents pick the same module simultaneously, the second one yields and picks the next unblocked module from the same Phase.

---

## 6. Coordination across modules

- Cross-module questions → open a GitHub Discussion under "Architecture".
- Protocol changes → open a PR under `core/protocols.py` first; once merged, dependent modules can adopt the new contract.
- For major design pivots, edit `docs/ARCHITECTURE.md` in the same PR.

---

## 7. Quality gates

A PR is mergeable when:
- [ ] All acceptance tests in your SPEC pass
- [ ] No layering violation
- [ ] No new pip dependency that isn't already in `pyproject.toml` (else justify in PR description and bump it)
- [ ] Code coverage for the new module ≥ 80%
- [ ] Ruff clean
- [ ] No regressions in benchmark suite (if your module has perf budget)

---

## 8. What if a Protocol is wrong?

Don't silently bend it. Open a PR titled `protocols: <reason>` that:
1. Updates `core/protocols.py`
2. Updates `docs/INTERFACES.md`
3. Updates **all** in-tree implementations
4. Bumps the protocol version comment block

Get one human review before merging Protocol PRs.

---

## 9. Recommended environment

```bash
./scripts/bootstrap_dev.sh        # creates .venv, installs deps, sets up pre-commit
source .venv/bin/activate
pytest -q                          # baseline must be green before you start
```

Cloud agents in headless containers can use the same script — it's CI-equivalent.

---

## 10. Where to ask for help

| Question | Channel |
| --- | --- |
| "Is this the right Protocol?" | GitHub Discussions → Architecture |
| "Is this the right algorithm?" | GitHub Discussions → Research |
| "My acceptance test is unclear" | Comment on your SPEC's source PR |
| "I need a new dependency" | Open an issue tagged `dependency-bump` |

Happy hacking.
