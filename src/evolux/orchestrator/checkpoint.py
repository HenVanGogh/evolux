"""Checkpoint save / load for evolution runs.

A checkpoint captures enough state to resume an :class:`AssembledRun`
bit-exactly:

* ``step`` — number of completed generations.
* population genomes (serialised via :meth:`~evolux.core.protocols.Genome.serialize`).
* population fitness tensor.

The loop uses deterministic per-generation RNG splits
(``rng.split(f"gen_{gen}")`` and ``rng.split(f"evo_{gen}")``) so resuming
from ``step = N`` automatically replays the same random stream for all
subsequent generations without saving any generator state.

Usage::

    from evolux.orchestrator import checkpoint

    checkpoint.save(run, "runs/exp1/checkpoints/gen_50.pkl")
    checkpoint.restore(run, "runs/exp1/checkpoints/gen_50.pkl")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from evolux.orchestrator.assemble import AssembledRun

log = logging.getLogger(__name__)


# ── Data container ────────────────────────────────────────────────────────────


@dataclass
class CheckpointData:
    """Data returned by :func:`load`.

    Attributes
    ----------
    step:
        Number of completed generations at save time.
    genomes:
        Deserialised :class:`~evolux.core.protocols.Genome` objects.
    fitness:
        Per-genome fitness tensor, shape ``(P,)``, float32.
    """

    step: int
    genomes: list[Any]
    fitness: Tensor


# ── Public API ────────────────────────────────────────────────────────────────


def save(run: AssembledRun, path: Path | str) -> None:
    """Serialise population state and step counter to *path*.

    The checkpoint is a ``pickle`` dict containing:

    * ``step`` — int
    * ``genomes`` — list of ``bytes`` (one per genome via
      :meth:`~evolux.core.protocols.Genome.serialize`)
    * ``fitness`` — ``numpy.ndarray`` float32, shape ``(P,)``
    * ``genome_encoding`` — str key for the deserialiser

    Parameters
    ----------
    run:
        Run whose population should be saved.
    path:
        Destination file path.  Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pop = run.population
    encoding: str = pop.genomes[0].encoding if pop.genomes else "direct"

    data: dict[str, Any] = {
        "step": run.step,
        "genomes": [g.serialize() for g in pop.genomes],
        "fitness": pop.fitness.cpu().numpy(),
        "genome_encoding": encoding,
    }

    with path.open("wb") as fh:
        pickle.dump(data, fh, protocol=4)

    log.info("checkpoint saved  path=%s  step=%d", path, run.step)


def load(path: Path | str) -> CheckpointData:
    """Deserialise a checkpoint from *path*.

    Parameters
    ----------
    path:
        Checkpoint file written by :func:`save`.

    Returns
    -------
    CheckpointData
        Deserialised step, genomes, and fitness.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    NotImplementedError
        If the genome encoding stored in the checkpoint is not yet supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with path.open("rb") as fh:
        data: dict[str, Any] = pickle.load(fh)  # noqa: S301

    encoding: str = data.get("genome_encoding", "direct")
    if encoding == "direct":
        from evolux.genome.direct import DirectGenome

        genomes: list[Any] = [DirectGenome.deserialize(b) for b in data["genomes"]]
    else:
        raise NotImplementedError(f"Unsupported genome encoding: {encoding!r}")

    import numpy as np

    raw: np.ndarray = data["fitness"]
    fitness = torch.from_numpy(raw.copy())

    log.info("checkpoint loaded  path=%s  step=%d", path, data["step"])
    return CheckpointData(step=data["step"], genomes=genomes, fitness=fitness)


def restore(run: AssembledRun, path: Path | str) -> None:
    """Restore *run*'s population and step counter from *path* (in-place).

    After this call ``run.step``, ``run.population.genomes``, and
    ``run.population.fitness`` reflect the saved state.  The run's RNG, brain,
    and world are untouched — they are deterministic from the seed and the
    restored step counter.

    Parameters
    ----------
    run:
        Run to restore into.
    path:
        Checkpoint file written by :func:`save`.
    """
    ckpt = load(path)
    run.step = ckpt.step
    run.population.genomes = ckpt.genomes
    run.population.fitness = ckpt.fitness.to(run.device)
    log.info("checkpoint restored  path=%s  step=%d", path, run.step)
