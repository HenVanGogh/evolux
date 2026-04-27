#!/usr/bin/env python3
"""Static layering check.

Enforces the dependency graph defined in docs/ARCHITECTURE.md.

Each module may import only from its own layer or lower layers (and stdlib /
third-party packages). Violations are printed and the script exits non-zero.

Usage:
    python scripts/check_layering.py [--verbose]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# ── Layer graph ─────────────────────────────────────────────────────────────
# Higher number = higher layer. A module may import from any module with
# layer <= its own.
LAYER: dict[str, int] = {
    "core": 0,
    "tensors": 0,
    "perception": 1,
    "memory": 1,
    "genome": 1,
    "brain": 2,
    "morphology": 2,
    "physics": 3,
    "world": 3,
    "environment": 4,
    "fitness": 4,
    "evolution": 5,
    "distributed": 6,
    "viz": 6,
    "orchestrator": 7,
}

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "evolux"


def module_of(path: Path) -> str | None:
    """Return the top-level module name (e.g. 'brain') for a file under src/evolux/."""
    try:
        rel = path.relative_to(SRC)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    return parts[0]


def imports_in(file: Path) -> set[str]:
    """Return set of evolux submodule names imported by *file* (e.g. 'brain', 'memory')."""
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith("evolux."):
                    found.add(n.name.split(".")[1])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("evolux."):
                found.add(node.module.split(".")[1])
            elif node.module == "evolux":
                pass
            # Relative imports
            if node.level >= 1 and node.module is None:
                # `from . import x` — no module hop
                pass
    return found


def main() -> int:
    verbose = "--verbose" in sys.argv
    if not SRC.exists():
        print(f"warn: {SRC} does not exist yet — skipping layering check.")
        return 0

    violations: list[str] = []
    for py in SRC.rglob("*.py"):
        mod = module_of(py)
        if mod is None or mod not in LAYER:
            continue
        own_layer = LAYER[mod]
        for dep in imports_in(py):
            if dep == mod or dep not in LAYER:
                continue
            if LAYER[dep] > own_layer:
                rel = py.relative_to(ROOT)
                violations.append(
                    f"{rel}: '{mod}' (layer {own_layer}) imports '{dep}' (layer {LAYER[dep]})"
                )
            elif verbose:
                print(f"ok  {py.relative_to(ROOT)}: {mod} -> {dep}")

    if violations:
        print("Layering violations:")
        for v in violations:
            print(f"  - {v}")
        print(f"\n{len(violations)} violation(s). See docs/ARCHITECTURE.md.")
        return 1

    print("Layering OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
