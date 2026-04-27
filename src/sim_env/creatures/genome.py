"""Genome: variable-length encoding of brain topology + body parameters.

A genome is split into two sections:
  - ``node_genes``       — neurons (input / hidden / output / memory)
  - ``connection_genes`` — directed weighted edges between neurons

This mirrors the NEAT encoding scheme, supporting:
  - historical innovation numbers for crossover alignment
  - add/remove node and connection mutations
  - weight perturbation mutations
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np


class NodeType(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()
    MEMORY_READ = auto()   # reads from episodic memory
    MEMORY_WRITE = auto()  # writes to episodic memory
    BIAS = auto()


class Activation(Enum):
    TANH = auto()
    RELU = auto()
    SIGMOID = auto()
    LINEAR = auto()
    SIN = auto()


@dataclass
class NodeGene:
    node_id: int
    node_type: NodeType
    activation: Activation = Activation.TANH
    bias: float = 0.0
    # For memory nodes: which memory slot they address
    memory_slot: int = 0

    def copy(self) -> "NodeGene":
        return copy.copy(self)


@dataclass
class ConnectionGene:
    innov: int          # global innovation number (for crossover alignment)
    src: int            # source node_id
    dst: int            # destination node_id
    weight: float = 0.0
    enabled: bool = True
    recurrent: bool = False  # forward (False) or recurrent/backward (True)

    def copy(self) -> "ConnectionGene":
        return copy.copy(self)


# Global innovation counter (in a real multi-process run this needs synchronisation)
_INNOV_COUNTER: int = 0


def next_innov() -> int:
    global _INNOV_COUNTER
    _INNOV_COUNTER += 1
    return _INNOV_COUNTER


@dataclass
class Genome:
    """Full genetic encoding of a creature.

    ``node_genes``       is a dict keyed by node_id for O(1) lookup.
    ``connection_genes`` is a dict keyed by innov number.

    Body parameters are stored as a separate float array decoded by the Body.
    """

    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_genes: dict[int, NodeGene] = field(default_factory=dict)
    connection_genes: dict[int, ConnectionGene] = field(default_factory=dict)
    body_params: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    # Tracking
    generation_born: int = 0
    parent_ids: list[str] = field(default_factory=list)

    # ── Accessors ────────────────────────────────────────────────────────────

    def nodes_of_type(self, nt: NodeType) -> list[NodeGene]:
        return [n for n in self.node_genes.values() if n.node_type == nt]

    def enabled_connections(self) -> list[ConnectionGene]:
        return [c for c in self.connection_genes.values() if c.enabled]

    def n_hidden(self) -> int:
        return sum(1 for n in self.node_genes.values() if n.node_type == NodeType.HIDDEN)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def minimal(
        cls,
        n_inputs: int,
        n_outputs: int,
        rng: np.random.Generator,
        weight_range: tuple[float, float] = (-1.0, 1.0),
    ) -> "Genome":
        """Create the minimal genome: inputs fully connected to outputs, no hidden."""
        g = cls()
        node_id = 0

        # Bias node
        bias_node = NodeGene(node_id, NodeType.BIAS, Activation.LINEAR)
        g.node_genes[node_id] = bias_node
        node_id += 1

        in_ids = []
        for _ in range(n_inputs):
            n = NodeGene(node_id, NodeType.INPUT, Activation.LINEAR)
            g.node_genes[node_id] = n
            in_ids.append(node_id)
            node_id += 1

        out_ids = []
        for _ in range(n_outputs):
            n = NodeGene(node_id, NodeType.OUTPUT, Activation.TANH)
            g.node_genes[node_id] = n
            out_ids.append(node_id)
            node_id += 1

        lo, hi = weight_range
        for src in in_ids + [0]:  # bias → outputs too
            for dst in out_ids:
                c = ConnectionGene(
                    innov=next_innov(),
                    src=src,
                    dst=dst,
                    weight=float(rng.uniform(lo, hi)),
                )
                g.connection_genes[c.innov] = c

        return g

    # ── Copy / clone ─────────────────────────────────────────────────────────

    def clone(self) -> "Genome":
        g = Genome(
            genome_id=str(uuid.uuid4())[:8],
            node_genes={k: v.copy() for k, v in self.node_genes.items()},
            connection_genes={k: v.copy() for k, v in self.connection_genes.items()},
            body_params=self.body_params.copy(),
            generation_born=self.generation_born,
            parent_ids=[self.genome_id],
        )
        return g

    # ── Compatibility distance (for speciation) ───────────────────────────────

    def compatibility_distance(
        self,
        other: "Genome",
        disjoint_coeff: float = 1.0,
        weight_coeff: float = 0.4,
    ) -> float:
        keys_self = set(self.connection_genes)
        keys_other = set(other.connection_genes)
        matching = keys_self & keys_other

        if not matching:
            disjoint = len(keys_self ^ keys_other)
            n = max(len(keys_self), len(keys_other), 1)
            return disjoint_coeff * disjoint / n

        w_diff = np.mean([
            abs(self.connection_genes[k].weight - other.connection_genes[k].weight)
            for k in matching
        ])
        disjoint = len(keys_self ^ keys_other)
        n = max(len(keys_self), len(keys_other), 1)
        return disjoint_coeff * disjoint / n + weight_coeff * w_diff

    def __repr__(self) -> str:
        return (
            f"Genome(id={self.genome_id}, "
            f"nodes={len(self.node_genes)}, "
            f"conns={len(self.connection_genes)})"
        )
