#!/usr/bin/env bash
# Bulk-create one "module-work" issue per module SPEC.
# Requires `gh` authenticated. Idempotent only if no matching titles exist yet.
set -euo pipefail

REPO="${REPO:-HenVanGogh/evolux}"

# Ensure required labels exist (best-effort).
for lbl in module-work phase:0 phase:1 phase:2 phase:3 phase:4; do
  gh label create "$lbl" --repo "$REPO" --color BFD4F2 --force >/dev/null 2>&1 || true
done

declare -A PHASES=(
  [core]=0 [tensors]=0
  [perception]=1 [memory]=1 [genome]=1 [brain]=1 [morphology]=1
  [physics]=1 [world]=1 [environment]=1 [fitness]=1 [evolution]=1
  [orchestrator]=1
  [distributed]=4 [viz]=1
)

for module in "${!PHASES[@]}"; do
  phase="${PHASES[$module]}"
  spec="src/evolux/${module}/SPEC.md"
  title="[${module}] Implement Phase ${phase} skeleton per SPEC"
  body=$(cat <<EOF
Implement the **Phase ${phase}** deliverables for the \`${module}\` module.

## Read first
- [\`AGENTS.md\`](AGENTS.md)
- [\`${spec}\`](${spec}) — files in scope, acceptance tests, perf budget.

## Scope
Only files under \`src/evolux/${module}/\` and \`tests/unit/${module}/\`.

## Done when
- All Phase ${phase} acceptance tests in the SPEC pass.
- \`ruff check . && ruff format --check .\` clean.
- \`python scripts/check_layering.py\` passes.
- PR follows the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
EOF
)
  echo ">> creating issue for ${module} (phase ${phase})"
  gh issue create \
    --repo "$REPO" \
    --title "$title" \
    --body "$body" \
    --label "module-work" \
    --label "phase:${phase}"
done
