"""Assemble an :class:`AssembledRun` from a typed config.

``assemble_from_config`` is the single entry point that reads
``cfg.modules.*`` registry keys, instantiates every component, and
returns a ready-to-run :class:`AssembledRun` dataclass.

When a module config key is absent a sensible Phase-1 default is used so
that ``configs/experiments/smoke.yaml`` (which has ``modules: {}``) can run
end-to-end without any additional configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from evolux.brain import BRAIN_REGISTRY
from evolux.brain.transformer import TransformerBrain
from evolux.core.config import EvoluxConfig
from evolux.core.device import get_device
from evolux.core.rng import RNG
from evolux.core.types import ActionSpec, Obs, ObsSpec
from evolux.evolution import EVOLUTION_REGISTRY
from evolux.evolution.population import Population
from evolux.evolution.tournament import Tournament
from evolux.fitness.aggregators import WeightedSum
from evolux.fitness.objectives import SurvivalTime
from evolux.genome import GENOME_OPERATOR_REGISTRY
from evolux.genome.direct import DirectGenome, DirectOperator
from evolux.viz.loggers import JsonlLogger
from evolux.world import WORLD_REGISTRY

log = logging.getLogger(__name__)

# ── Default specs used when no world is configured ────────────────────────────

#: Four-dimensional continuous observation space used by the stub world.
DEFAULT_OBS_SPEC: ObsSpec = ObsSpec(fields={"obs": ((4,), torch.float32)})

#: Two-dimensional continuous action space used by the stub world.
DEFAULT_ACTION_SPEC: ActionSpec = ActionSpec(discrete=False, n=2, bounds=(-1.0, 1.0))


# ── Stub world ────────────────────────────────────────────────────────────────


class _StubWorld:
    """Minimal world stub for Phase-1 smoke runs.

    Returns random observations and zero rewards.  Used when no ``world``
    registry key is present in ``cfg.modules``.

    Each generation's observations are deterministic from ``(seed, gen)`` so
    that checkpoint-restored runs reproduce the exact same trajectory.
    """

    def __init__(
        self,
        batch_size: int,
        obs_spec: ObsSpec,
        device: torch.device,
        rng: RNG,
    ) -> None:
        self.batch_size: int = batch_size
        self.device: torch.device = device
        self._obs_spec: ObsSpec = obs_spec
        self._base_rng: RNG = rng
        # Initialised to a valid generator; overwritten by reset().
        self._gen: torch.Generator = rng.split("stub_world_init")
        self._current_gen: int = 0

    def _prepare_generation(self, gen: int) -> None:
        """Record the upcoming generation index for deterministic RNG seeding.

        Called by :class:`EvolutionLoop` before each ``reset()`` call so that
        observations for generation *gen* are always reproducible regardless of
        how many previous generations have run.
        """
        self._current_gen = gen

    def reset(self, mask: Tensor | None = None) -> None:
        """Create a fresh per-generation generator (ignores *mask*)."""
        self._gen = self._base_rng.split(f"stub_world_gen_{self._current_gen}")

    def observe(self) -> Obs:
        """Return standard-normal observations matching the declared :class:`ObsSpec`."""
        result: Obs = {}
        for name, (shape, dtype) in self._obs_spec.fields.items():
            t = torch.zeros(self.batch_size, *shape, device=self.device)
            t.normal_(generator=self._gen)
            if dtype != torch.float32:
                t = t.to(dtype)
            result[name] = t
        return result

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Return zero rewards and no terminations (stub has no dynamics)."""
        reward = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        return reward, done, {}


# ── AssembledRun ─────────────────────────────────────────────────────────────


@dataclass
class AssembledRun:
    """All components wired together for a single evolution run.

    Created by :func:`assemble_from_config`.  Passed to
    :class:`~evolux.orchestrator.loop.EvolutionLoop` and
    :func:`~evolux.orchestrator.checkpoint.save` /
    :func:`~evolux.orchestrator.checkpoint.restore`.

    Attributes
    ----------
    world:
        A :class:`~evolux.core.protocols.World` implementation (or
        :class:`_StubWorld` in smoke-test mode).
    brain:
        A :class:`~evolux.core.protocols.Brain` implementation.
    population:
        A :class:`~evolux.evolution.population.Population` holding genomes and
        per-genome fitness scores.
    selector:
        A :class:`~evolux.core.protocols.Selector` implementation.
    operator:
        A :class:`~evolux.core.protocols.GenomeOperator` implementation.
    aggregator:
        A :class:`~evolux.core.protocols.FitnessAggregator` implementation.
    logger:
        A :class:`~evolux.core.protocols.StatsLogger` implementation.
    rng:
        Root :class:`~evolux.core.rng.RNG` for this run.
    run_dir:
        Directory where logs, checkpoints, and the config copy live.
    cfg:
        Fully-resolved experiment config.
    device:
        Resolved :class:`torch.device`.
    step:
        Number of completed generations (mutated in-place by
        :class:`~evolux.orchestrator.loop.EvolutionLoop`).
    """

    world: Any
    brain: Any
    population: Any
    selector: Any
    operator: Any
    aggregator: Any
    logger: Any
    rng: RNG
    run_dir: Path
    cfg: EvoluxConfig
    device: torch.device
    step: int = 0

    def close(self) -> None:
        """Close any open resources (e.g. the stats logger file handle).

        Idempotent — safe to call multiple times.
        """
        if self.logger is not None:
            self.logger.close()


# ── Private builder helpers ───────────────────────────────────────────────────


def _build_world(
    cfg: EvoluxConfig,
    obs_spec: ObsSpec,
    device: torch.device,
    rng: RNG,
) -> Any:
    """Instantiate a World from ``cfg.modules.world``, or a stub."""
    world_cfg: dict[str, Any] = cfg.modules.get("world", {})  # type: ignore[assignment]
    if world_cfg:
        world_type: str = world_cfg.get("type", "")
        world_cls = WORLD_REGISTRY.get(world_type)
        log.info("world=%s", world_type)
        return world_cls(**{k: v for k, v in world_cfg.items() if k != "type"})
    log.info("world=_StubWorld (no modules.world configured)")
    return _StubWorld(
        batch_size=cfg.simulation.batch_size,
        obs_spec=obs_spec,
        device=device,
        rng=rng.child("world"),
    )


def _build_brain(
    obs_spec: ObsSpec,
    action_spec: ActionSpec,
    device: torch.device,
    brain_cfg: dict[str, Any],
) -> Any:
    """Instantiate a Brain from *brain_cfg*, or a small default Transformer."""
    if brain_cfg:
        brain_type: str = brain_cfg.get("type", "transformer")
        brain_cls = BRAIN_REGISTRY.get(brain_type)
        log.info("brain=%s", brain_type)
        brain = brain_cls(
            obs_spec,
            action_spec,
            **{k: v for k, v in brain_cfg.items() if k != "type"},
        )
    else:
        log.info("brain=TransformerBrain (default)")
        brain = TransformerBrain(
            obs_spec,
            action_spec,
            hidden_dim=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=16,
        )
    return brain.to(device)


def _build_population_and_operator(
    cfg: EvoluxConfig,
    brain: Any,
    device: torch.device,
    rng: RNG,
    genome_cfg: dict[str, Any],
) -> tuple[Any, Any]:
    """Build a Population of DirectGenomes + a GenomeOperator."""
    if genome_cfg:
        op_type: str = genome_cfg.get("operator_type", genome_cfg.get("type", "direct"))
        op_cls = GENOME_OPERATOR_REGISTRY.get(op_type)
        operator: Any = op_cls(**genome_cfg.get("operator", {}))
        log.info("operator=%s", op_type)
    else:
        operator = DirectOperator(sigma=0.01)
        log.info("operator=DirectOperator (default)")

    n_params = sum(int(p.numel()) for p in brain.parameters())
    init_gen = rng.split("init_genomes")
    genomes = [
        DirectGenome(torch.rand(n_params, generator=init_gen)) for _ in range(cfg.population.size)
    ]
    return Population.from_genomes(genomes, device=device), operator


def _build_selector(evo_cfg: dict[str, Any]) -> Any:
    """Instantiate a Selector from *evo_cfg*, or Tournament with defaults."""
    if evo_cfg:
        evo_type: str = evo_cfg.get("type", "tournament")
        selector_cls = EVOLUTION_REGISTRY.get(evo_type)
        log.info("selector=%s", evo_type)
        return selector_cls(**{k: v for k, v in evo_cfg.items() if k != "type"})
    log.info("selector=Tournament (default)")
    return Tournament(k=3, n_elites=1)


# ── Builder ───────────────────────────────────────────────────────────────────


def assemble_from_config(
    cfg: EvoluxConfig,
    run_dir: Path | None = None,
) -> AssembledRun:
    """Build an :class:`AssembledRun` from *cfg*.

    Reads ``cfg.modules.{world, brain, genome, evolution, fitness}`` registry
    keys and instantiates the corresponding components.  When a key is absent
    the function falls back to a sensible Phase-1 default so that
    ``configs/experiments/smoke.yaml`` (``modules: {}``) works out-of-the-box.

    Parameters
    ----------
    cfg:
        Validated config (from :func:`evolux.core.config.load_config`).
    run_dir:
        Output directory.  If *None* a timestamped subdirectory of
        ``cfg.simulation.output_dir`` is created automatically.

    Returns
    -------
    AssembledRun
        Fully-initialised run container ready for
        :class:`~evolux.orchestrator.loop.EvolutionLoop`.
    """
    device = get_device(cfg.simulation.device)
    log.info("device=%s", device)

    if run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.simulation.output_dir) / run_id
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write the fully-resolved config for reproducibility.
    config_yaml = yaml.dump(cfg.model_dump(), default_flow_style=False, sort_keys=False)
    (run_dir / "config.yaml").write_text(config_yaml, encoding="utf-8")
    log.info("run_dir=%s", run_dir)

    rng = RNG(cfg.simulation.seed, device=device)

    obs_spec: ObsSpec = DEFAULT_OBS_SPEC
    action_spec: ActionSpec = DEFAULT_ACTION_SPEC

    world = _build_world(cfg, obs_spec, device, rng)
    brain = _build_brain(
        obs_spec,
        action_spec,
        device,
        cfg.modules.get("brain", {}),  # type: ignore[arg-type]
    )
    population, operator = _build_population_and_operator(
        cfg,
        brain,
        device,
        rng,
        cfg.modules.get("genome", {}),  # type: ignore[arg-type]
    )
    selector = _build_selector(cfg.modules.get("evolution", {}))  # type: ignore[arg-type]

    fit_cfg = cfg.modules.get("fitness", {})
    if fit_cfg:
        log.info("fitness config present but registry wiring is Phase-2; using default")
    aggregator: Any = WeightedSum([SurvivalTime(weight=1.0)])
    log.info("aggregator=WeightedSum[SurvivalTime] (default)")

    logger: Any = JsonlLogger(run_dir / "logs" / "events.jsonl")

    return AssembledRun(
        world=world,
        brain=brain,
        population=population,
        selector=selector,
        operator=operator,
        aggregator=aggregator,
        logger=logger,
        rng=rng,
        run_dir=run_dir,
        cfg=cfg,
        device=device,
    )
