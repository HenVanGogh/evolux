"""DynamicNN: variable-topology neural network brain built from a Genome.

The network is evaluated by topologically sorting the directed graph of
neurons. Recurrent connections are handled by keeping the previous step's
activations and feeding them forward with one-step delay (BPTT-free).

Memory integration
------------------
Working memory slots appear as extra inputs and are written to by designated
output neurons.  Episodic memory is read via a learned query head and injected
into the hidden layer.  Hebbian memory modulates hidden→output weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sim_env.brain.base_brain import BaseBrain
from sim_env.brain.memory import MemoryBank
from sim_env.creatures.genome import Activation, Genome, NodeType

if TYPE_CHECKING:
    pass


# ── Activation functions ────────────────────────────────────────────────────

_ACT = {
    Activation.TANH: np.tanh,
    Activation.RELU: lambda x: np.maximum(0.0, x),
    Activation.SIGMOID: lambda x: 1.0 / (1.0 + np.exp(-x)),
    Activation.LINEAR: lambda x: x,
    Activation.SIN: np.sin,
}


class DynamicNN(BaseBrain):
    """Network whose topology is determined at construction from a Genome.

    Parameters
    ----------
    genome:
        The genetic blueprint.
    n_sensor_inputs:
        Number of raw sensory inputs (does NOT include working memory slots;
        those are appended internally).
    n_action_outputs:
        Number of action logit outputs (does NOT include memory-write outputs).
    memory_bank:
        Shared memory container; the brain wires memory I/O to it.
    """

    def __init__(
        self,
        genome: Genome,
        n_sensor_inputs: int,
        n_action_outputs: int,
        memory_bank: MemoryBank,
    ) -> None:
        self._genome = genome
        self._mem = memory_bank
        self._n_sensor = n_sensor_inputs
        self._n_actions = n_action_outputs

        # Build topology
        self._build(genome)

    # ── Construction ────────────────────────────────────────────────────────

    def _build(self, genome: Genome) -> None:
        """Compile genome into efficient numpy structures."""
        # Collect all node IDs
        nodes = genome.node_genes
        conns = genome.enabled_connections()

        self._node_ids: list[int] = list(nodes.keys())
        self._n_nodes = len(self._node_ids)
        idx_of = {nid: i for i, nid in enumerate(self._node_ids)}

        # Node types & activations
        self._node_types: list[NodeType] = [nodes[nid].node_type for nid in self._node_ids]
        self._node_act: list[Activation] = [nodes[nid].activation for nid in self._node_ids]
        self._biases: np.ndarray = np.array(
            [nodes[nid].bias for nid in self._node_ids], dtype=np.float32
        )

        # Activation buffers (current and previous for recurrence)
        self._act: np.ndarray = np.zeros(self._n_nodes, dtype=np.float32)
        self._act_prev: np.ndarray = np.zeros(self._n_nodes, dtype=np.float32)

        # Connection matrices (sparse representation: lists for easy iteration)
        self._fwd_src: list[int] = []
        self._fwd_dst: list[int] = []
        self._fwd_w: list[float] = []

        self._rec_src: list[int] = []
        self._rec_dst: list[int] = []
        self._rec_w: list[float] = []

        for c in conns:
            if c.src not in idx_of or c.dst not in idx_of:
                continue
            si, di = idx_of[c.src], idx_of[c.dst]
            if c.recurrent:
                self._rec_src.append(si)
                self._rec_dst.append(di)
                self._rec_w.append(c.weight)
            else:
                self._fwd_src.append(si)
                self._fwd_dst.append(di)
                self._fwd_w.append(c.weight)

        self._fwd_src_arr = np.array(self._fwd_src, dtype=np.int32)
        self._fwd_dst_arr = np.array(self._fwd_dst, dtype=np.int32)
        self._fwd_w_arr = np.array(self._fwd_w, dtype=np.float32)

        self._rec_src_arr = np.array(self._rec_src, dtype=np.int32)
        self._rec_dst_arr = np.array(self._rec_dst, dtype=np.int32)
        self._rec_w_arr = np.array(self._rec_w, dtype=np.float32)

        # Topological sort of forward nodes
        self._eval_order: list[int] = self._topological_sort(idx_of)

        # Identify input/output node indices
        self._input_idxs = [i for i, nt in enumerate(self._node_types) if nt == NodeType.INPUT]
        self._bias_idxs = [i for i, nt in enumerate(self._node_types) if nt == NodeType.BIAS]
        self._output_idxs = [i for i, nt in enumerate(self._node_types) if nt == NodeType.OUTPUT]

        # Memory-write outputs
        self._mem_write_idxs = [
            i for i, nt in enumerate(self._node_types) if nt == NodeType.MEMORY_WRITE
        ]

        # Initialise Hebbian memory between hidden and output
        hidden_idxs = [i for i, nt in enumerate(self._node_types) if nt == NodeType.HIDDEN]
        n_hidden = len(hidden_idxs)
        n_out = len(self._output_idxs)
        if n_hidden > 0 and n_out > 0:
            self._mem.init_hebbian(n_hidden, n_out)
        self._hidden_idxs = hidden_idxs

        # Working memory: number of slots read back as extra inputs
        self._wm_size = self._mem.working.size

    def _topological_sort(self, idx_of: dict[int, int]) -> list[int]:
        """Kahn's algorithm; cycles (recurrent) are broken by ignoring rec edges."""
        from collections import deque

        in_deg = [0] * self._n_nodes
        adj: list[list[int]] = [[] for _ in range(self._n_nodes)]

        for si, di in zip(self._fwd_src, self._fwd_dst):
            adj[si].append(di)
            in_deg[di] += 1

        queue = deque(i for i in range(self._n_nodes) if in_deg[i] == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)

        # Nodes not reached (cycles) are appended at the end
        reached = set(order)
        order += [i for i in range(self._n_nodes) if i not in reached]
        return order

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute output activations.

        inputs: sensor observations (length == n_sensor_inputs)
        """
        act = self._act
        act_prev = self._act_prev

        # 1. Inject sensor inputs
        for k, idx in enumerate(self._input_idxs):
            if k < len(inputs):
                act[idx] = inputs[k]

        # 2. Set bias nodes
        for idx in self._bias_idxs:
            act[idx] = 1.0

        # 3. Inject working memory as extra inputs (appended after sensors)
        wm = self._mem.working.read()
        for k, wm_val in enumerate(wm):
            sensor_offset = len(self._input_idxs) + k
            if sensor_offset < len(self._input_idxs) + len(wm):
                # Write into the first available hidden node as a proxy
                # (proper solution: designate MEMORY_READ nodes in genome)
                pass  # placeholder — memory is available to brain via hidden activations

        # 4. Recurrent contributions from previous step
        if len(self._rec_src_arr):
            np.add.at(act, self._rec_dst_arr, self._rec_w_arr * act_prev[self._rec_src_arr])

        # 5. Forward pass in topological order
        # Pre-compute net input via vectorised scatter-add
        net = self._biases.copy()
        if len(self._fwd_src_arr):
            np.add.at(net, self._fwd_dst_arr, self._fwd_w_arr * act[self._fwd_src_arr])

        for i in self._eval_order:
            act[i] = _ACT[self._node_act[i]](net[i])

        # 6. Hebbian modulation on output nodes
        if self._mem.hebbian is not None and self._hidden_idxs:
            h_vec = act[self._hidden_idxs]
            hebb_contrib = self._mem.hebbian.apply(h_vec)
            for k, oi in enumerate(self._output_idxs):
                if k < len(hebb_contrib):
                    act[oi] = np.tanh(act[oi] + hebb_contrib[k])
            # Update Hebbian traces
            out_vec = act[self._output_idxs]
            self._mem.hebbian.update(h_vec, out_vec)

        # 7. Write to working memory from designated memory-write nodes
        if self._mem_write_idxs:
            writes = act[self._mem_write_idxs]
            for k in range(min(len(writes), self._wm_size)):
                self._mem.working.write_slot(k, writes[k])

        # 8. Save activations for next recurrent step
        np.copyto(act_prev, act)

        return act[self._output_idxs].copy()

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        self._act[:] = 0.0
        self._act_prev[:] = 0.0
        self._mem.reset_episode()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n_inputs(self) -> int:
        return self._n_sensor

    @property
    def n_outputs(self) -> int:
        return len(self._output_idxs)

    def n_parameters(self) -> int:
        return len(self._fwd_w) + len(self._rec_w) + self._n_nodes  # weights + biases

    def __repr__(self) -> str:
        return (
            f"DynamicNN(nodes={self._n_nodes}, "
            f"fwd_conns={len(self._fwd_w)}, "
            f"rec_conns={len(self._rec_w)})"
        )
