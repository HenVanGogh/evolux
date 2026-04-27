# sim_env — Simulated Evolution Framework

A research-grade scaffold for evolving creatures with **dynamic brains**, **memory**, and **directed evolution** via goal gradients.

## Architecture

```
src/sim_env/
├── core/           # Simulation loop, world, registry
├── creatures/      # Creature, genome, body, sensors
├── brain/          # Dynamic NN topology, memory banks, NEAT-style growth
├── evolution/      # Population, selection, operators, directed evolution
├── environment/    # Grid/continuous environments, resources
├── fitness/        # Multi-objective evaluators, goal gradients
└── visualization/  # Renderer, stats logger
```

## Key Concepts

### Dynamic Brain
Each creature's brain is a directed graph of neurons whose size and wiring are encoded in the genome. Nodes carry activation state and optional recurrent memory. Topology grows/shrinks via NEAT-style structural mutations.

### Memory
Three tiers:
- **Working memory** — fixed-size activation buffer (fast, per-step)
- **Episodic memory** — key-value store written and queried by the brain
- **Structural memory** — slow-weight Hebbian traces that persist across lifetimes

### Directed Evolution
Beyond pure fitness maximization, creatures can be steered by:
- **Goal gradients** — scalar or vector field in environment space
- **Novelty + objective** — MAP-Elites or NS-ES style diversity pressure
- **Evolutionary gradient estimation** — finite-difference gradient approximation over population fitness

## Quick Start

```bash
pip install -e ".[dev,vis]"
python -m sim_env.cli run --config config/default.yaml
```

## Running Experiments

```bash
python experiments/run_experiment.py --config config/experiments/foraging.yaml
```
