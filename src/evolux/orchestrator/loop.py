"""Generation-based evolution loop.

:class:`EvolutionLoop` drives the canonical training loop described in
``docs/ARCHITECTURE.md``:

1. Decode genomes → brains *(Phase 2; currently uses a shared default brain)*.
2. Reset world & place creatures.
3. Roll out ``steps_per_generation`` steps collecting a :class:`~evolux.core.types.Trajectory`.
4. Compute multi-objective fitness and aggregate to a ``(B,)`` score.
5. Update population fitness scores.
6. Select parents, reproduce, mutate (one generation forward).
7. Log scalars; optionally write a checkpoint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from evolux.core.types import Obs, Trajectory

if TYPE_CHECKING:
    from evolux.orchestrator.assemble import AssembledRun

log = logging.getLogger(__name__)


class EvolutionLoop:
    """Drives the evolution loop for an :class:`~evolux.orchestrator.assemble.AssembledRun`.

    Parameters
    ----------
    run:
        Fully-assembled run produced by
        :func:`~evolux.orchestrator.assemble.assemble_from_config`.
    """

    def __init__(self, run: AssembledRun) -> None:
        self._run: AssembledRun = run

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, max_generations: int | None = None) -> None:
        """Execute the evolution loop.

        Parameters
        ----------
        max_generations:
            Number of generations to run.  If *None*, uses
            ``cfg.simulation.max_generations``.  Can be called multiple times
            on the same :class:`AssembledRun` to continue from where it left
            off (``run.step`` is incremented accordingly).
        """
        run = self._run
        cfg = run.cfg
        n_gen = max_generations if max_generations is not None else cfg.simulation.max_generations

        for i in range(n_gen):
            gen = run.step + i  # 0-based absolute generation index

            # ── Evaluate current population ───────────────────────────────────
            trajectory = self._evaluate(gen)

            # ── Compute fitness ───────────────────────────────────────────────
            scores: dict[str, Tensor] = {
                obj.name: obj.evaluate(trajectory, run.world) for obj in run.aggregator.objectives
            }
            fitness = run.aggregator.aggregate(scores)  # (B,)

            # ── Update population fitness (clip to pop size if B > P) ─────────
            p_size = len(run.population.genomes)
            if p_size > 0:
                run.population.fitness = fitness[:p_size].clone()

            # ── Evolve to next generation ─────────────────────────────────────
            evo_rng = run.rng.split(f"evo_{gen}")
            run.population.step_generation(run.selector, run.operator, evo_rng)

            # ── Logging ───────────────────────────────────────────────────────
            if (gen + 1) % cfg.simulation.log_interval == 0:
                mean_fit = fitness.mean().item()
                max_fit = fitness.max().item()
                run.logger.log_scalar("fitness/mean", mean_fit, gen + 1)
                run.logger.log_scalar("fitness/max", max_fit, gen + 1)
                log.info(
                    "gen=%d  fitness/mean=%.4f  fitness/max=%.4f",
                    gen + 1,
                    mean_fit,
                    max_fit,
                )

        run.step += n_gen

    # ── internals ─────────────────────────────────────────────────────────────

    def _evaluate(self, gen: int) -> Trajectory:
        """Roll out one generation and return the collected :class:`Trajectory`.

        Parameters
        ----------
        gen:
            Absolute (0-based) generation index.  Used to seed per-generation
            random sub-streams so checkpoint-restored runs reproduce the same
            trajectory.

        Returns
        -------
        Trajectory
            Stacked trajectory of shape ``(B, T, ...)``.
        """
        run = self._run
        cfg = run.cfg
        b = cfg.simulation.batch_size
        t_steps = cfg.simulation.steps_per_generation
        device = run.device

        # Notify stub world (and any world that supports it) of the upcoming gen.
        if hasattr(run.world, "prepare_generation"):
            run.world.prepare_generation(gen)
        run.world.reset()

        # Initialise brain recurrent state for the whole batch.
        brain_state = run.brain.init_state(b, device)

        obs_list: list[Obs] = []
        action_list: list[Tensor] = []
        reward_list: list[Tensor] = []
        done_list: list[Tensor] = []

        for _ in range(t_steps):
            obs = run.world.observe()
            action, brain_state, _ = run.brain.forward(obs, brain_state)
            reward, done, _ = run.world.step(action)

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)

        # Stack each field along the time dimension.
        stacked_obs: Obs = {k: torch.stack([o[k] for o in obs_list], dim=1) for k in obs_list[0]}

        return Trajectory(
            obs=stacked_obs,
            actions=torch.stack(action_list, dim=1),
            rewards=torch.stack(reward_list, dim=1),
            dones=torch.stack(done_list, dim=1),
            length=torch.full((b,), t_steps, dtype=torch.long, device=device),
        )
