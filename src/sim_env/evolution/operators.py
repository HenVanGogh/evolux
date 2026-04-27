"""Mutation and crossover operators for genomes."""

from __future__ import annotations

import numpy as np

from sim_env.creatures.genome import (
    Activation,
    ConnectionGene,
    Genome,
    NodeGene,
    NodeType,
    next_innov,
)


_ACTIVATIONS = list(Activation)


class MutationOperator:
    """Applies structural and parametric mutations to a genome copy.

    All probabilities are read from the ``evolution.operators`` config section.
    """

    def __init__(self, cfg: dict) -> None:
        op = cfg["evolution"]["operators"]
        self.w_mut_rate = op.get("weight_mutation_rate", 0.8)
        self.w_mut_sigma = op.get("weight_mutation_sigma", 0.2)
        self.w_perturb_rate = op.get("weight_perturbation_rate", 0.9)
        self.add_node_rate = op.get("add_node_rate", 0.03)
        self.del_node_rate = op.get("delete_node_rate", 0.01)
        self.add_conn_rate = op.get("add_connection_rate", 0.05)
        self.del_conn_rate = op.get("delete_connection_rate", 0.02)
        self.crossover_rate = op.get("crossover_rate", 0.75)
        self._weight_range = cfg["genome"].get("weight_range", [-3.0, 3.0])
        self._max_hidden = cfg["brain"].get("max_hidden", 64)
        self._min_hidden = cfg["brain"].get("min_hidden", 4)

    # ── Main entry ────────────────────────────────────────────────────────────

    def mutate(self, genome: Genome, rng: np.random.Generator) -> Genome:
        """Return a mutated clone of *genome*."""
        g = genome.clone()

        self._mutate_weights(g, rng)
        self._mutate_body_params(g, rng)

        if rng.random() < self.add_conn_rate:
            self._add_connection(g, rng)
        if rng.random() < self.del_conn_rate:
            self._disable_connection(g, rng)
        if rng.random() < self.add_node_rate and g.n_hidden() < self._max_hidden:
            self._add_node(g, rng)
        if rng.random() < self.del_node_rate and g.n_hidden() > self._min_hidden:
            self._delete_node(g, rng)

        return g

    def crossover(
        self, parent_a: Genome, parent_b: Genome, rng: np.random.Generator
    ) -> Genome:
        """Uniform crossover aligned by innovation number.

        The more-fit parent should be ``parent_a`` (its disjoint/excess genes
        are inherited by default).
        """
        child = parent_a.clone()
        child.parent_ids = [parent_a.genome_id, parent_b.genome_id]

        for innov, conn_b in parent_b.connection_genes.items():
            if innov in child.connection_genes:
                # Matching gene: pick randomly
                if rng.random() < 0.5:
                    child.connection_genes[innov] = conn_b.copy()
            # Disjoint/excess from parent_b are not added (parent_a dominates)

        # Body params blend
        alpha = rng.random(len(parent_a.body_params))
        child.body_params = (alpha * parent_a.body_params + (1 - alpha) * parent_b.body_params)

        return child

    # ── Weight mutations ──────────────────────────────────────────────────────

    def _mutate_weights(self, g: Genome, rng: np.random.Generator) -> None:
        lo, hi = self._weight_range
        for c in g.connection_genes.values():
            if rng.random() < self.w_mut_rate:
                if rng.random() < self.w_perturb_rate:
                    c.weight = float(np.clip(
                        c.weight + rng.normal(0, self.w_mut_sigma), lo, hi
                    ))
                else:
                    c.weight = float(rng.uniform(lo, hi))

        for n in g.node_genes.values():
            if rng.random() < self.w_mut_rate * 0.3:
                n.bias = float(np.clip(n.bias + rng.normal(0, self.w_mut_sigma), lo, hi))

    def _mutate_body_params(self, g: Genome, rng: np.random.Generator) -> None:
        mask = rng.random(len(g.body_params)) < 0.3
        g.body_params[mask] += rng.normal(0, 0.3, size=int(mask.sum())).astype(np.float32)

    # ── Structural mutations ─────────────────────────────────────────────────

    def _add_connection(self, g: Genome, rng: np.random.Generator) -> None:
        node_ids = list(g.node_genes.keys())
        if len(node_ids) < 2:
            return
        src, dst = rng.choice(node_ids, size=2, replace=False)
        # Don't connect to input/bias nodes as dst
        dst_node = g.node_genes[dst]
        if dst_node.node_type in (NodeType.INPUT, NodeType.BIAS):
            return
        # Don't duplicate existing forward connections
        existing = {(c.src, c.dst) for c in g.connection_genes.values() if not c.recurrent}
        lo, hi = self._weight_range
        recurrent = bool(rng.random() < 0.1)  # 10% chance of recurrent
        if not recurrent and (src, dst) in existing:
            return
        c = ConnectionGene(
            innov=next_innov(),
            src=src,
            dst=dst,
            weight=float(rng.uniform(lo, hi)),
            recurrent=recurrent,
        )
        g.connection_genes[c.innov] = c

    def _disable_connection(self, g: Genome, rng: np.random.Generator) -> None:
        enabled = [k for k, c in g.connection_genes.items() if c.enabled]
        if not enabled:
            return
        k = int(rng.choice(enabled))
        g.connection_genes[k].enabled = False

    def _add_node(self, g: Genome, rng: np.random.Generator) -> None:
        """Split an existing connection and insert a new hidden node."""
        enabled = [c for c in g.connection_genes.values() if c.enabled]
        if not enabled:
            return
        conn = enabled[int(rng.integers(len(enabled)))]
        conn.enabled = False

        new_id = max(g.node_genes.keys()) + 1
        act = _ACTIVATIONS[int(rng.integers(len(_ACTIVATIONS)))]
        new_node = NodeGene(new_id, NodeType.HIDDEN, activation=act)
        g.node_genes[new_id] = new_node

        # New connections: src → new_node (weight=1) and new_node → dst (original weight)
        g.connection_genes[next_innov()] = ConnectionGene(
            innov=next_innov(), src=conn.src, dst=new_id, weight=1.0
        )
        g.connection_genes[next_innov()] = ConnectionGene(
            innov=next_innov(), src=new_id, dst=conn.dst, weight=conn.weight
        )

    def _delete_node(self, g: Genome, rng: np.random.Generator) -> None:
        """Remove a random hidden node and all its connections."""
        hidden = [nid for nid, n in g.node_genes.items() if n.node_type == NodeType.HIDDEN]
        if not hidden:
            return
        target = int(rng.choice(hidden))
        del g.node_genes[target]
        to_del = [k for k, c in g.connection_genes.items() if c.src == target or c.dst == target]
        for k in to_del:
            del g.connection_genes[k]
