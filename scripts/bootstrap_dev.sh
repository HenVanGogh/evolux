#!/usr/bin/env bash
# Bootstrap a development environment for evolux.
# Idempotent: safe to re-run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
    echo ">>> Creating venv at .venv"
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo ">>> Upgrading pip"
pip install --upgrade pip wheel setuptools -q

echo ">>> Installing evolux in editable mode with [dev] extras"
pip install -e ".[dev]" -q

# Optional extras: install if requested via env var
if [ "${INSTALL_VIZ:-0}" = "1" ]; then
    pip install -e ".[viz]" -q
fi
if [ "${INSTALL_DISTRIBUTED:-0}" = "1" ]; then
    pip install -e ".[distributed]" -q
fi
if [ "${INSTALL_RETRIEVAL:-0}" = "1" ]; then
    pip install -e ".[retrieval]" -q
fi
if [ "${INSTALL_WEB:-0}" = "1" ]; then
    pip install -e ".[web]" -q
fi

echo ">>> Installing pre-commit hooks"
pre-commit install --install-hooks 2>/dev/null || true

echo ">>> Done."
echo ">>> Activate with:    source .venv/bin/activate"
echo ">>> Run tests with:   pytest -q"
