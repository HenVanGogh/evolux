---
applyTo: "**/SPEC.md"
---

# SPEC.md format guidelines

Every `src/evolux/<module>/SPEC.md` follows the same structure:

1. **Header table** with: Layer, Phase, Depends on, Exposes, Implements (Protocols).
2. **Mission** — a single paragraph: what this module is responsible for and what it deliberately avoids.
3. **Submodules** table — one row per file you must deliver.
4. **Common contract** — code block showing the Protocol(s) implemented.
5. **Acceptance tests** — bullet list of testable assertions. Each maps to one or more `tests/unit/<module>/test_*.py` cases.
6. **Performance budget** — concrete ms/throughput numbers on a stated reference machine.
7. **References** (optional) — papers / prior art.

Keep entries short and precise. SPEC.md is a contract, not documentation.
