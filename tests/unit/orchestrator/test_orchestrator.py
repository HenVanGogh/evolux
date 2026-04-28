"""Unit tests for evolux.orchestrator — Phase 1 acceptance tests.

Acceptance criteria (from SPEC):
- ``evolux validate configs/experiments/smoke.yaml`` exits 0.
- ``evolux run`` produces a ``runs/<id>/`` directory with config copy + at
  least one log entry.
- ``checkpoint.save → load`` is bit-exact (same RNG state → same evolutionary
  trajectory).
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

from evolux.core.config import EvoluxConfig, PopulationConfig, SimulationConfig
from evolux.orchestrator import checkpoint
from evolux.orchestrator.assemble import AssembledRun, assemble_from_config
from evolux.orchestrator.loop import EvolutionLoop

# ── Shared helpers ────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parents[3]


def make_smoke_cfg(seed: int = 42) -> EvoluxConfig:
    """Minimal config for fast unit tests (4 agents x 2 gens x 8 steps)."""
    return EvoluxConfig(
        simulation=SimulationConfig(
            seed=seed,
            device="cpu",
            batch_size=4,
            max_generations=2,
            steps_per_generation=8,
            log_interval=1,
            checkpoint_interval=100,  # prevent auto-checkpointing during tests
            output_dir="runs",
        ),
        population=PopulationConfig(size=4),
    )


@pytest.fixture
def smoke_cfg() -> EvoluxConfig:
    return make_smoke_cfg()


@pytest.fixture
def assembled(smoke_cfg: EvoluxConfig, tmp_path: Path) -> Generator[AssembledRun, None, None]:
    run = assemble_from_config(smoke_cfg, run_dir=tmp_path / "run")
    yield run
    run.close()


# ── assemble_from_config ──────────────────────────────────────────────────────


def test_assemble_creates_run_dir(smoke_cfg: EvoluxConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "my_run"
    run = assemble_from_config(smoke_cfg, run_dir=run_dir)
    run.close()
    assert run_dir.is_dir()


def test_assemble_writes_config_copy(smoke_cfg: EvoluxConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run = assemble_from_config(smoke_cfg, run_dir=run_dir)
    run.close()
    assert (run_dir / "config.yaml").is_file()
    content = (run_dir / "config.yaml").read_text()
    assert len(content) > 0


def test_assemble_returns_all_components(assembled: AssembledRun) -> None:
    assert assembled.world is not None
    assert assembled.brain is not None
    assert assembled.population is not None
    assert assembled.selector is not None
    assert assembled.operator is not None
    assert assembled.aggregator is not None
    assert assembled.logger is not None


def test_assemble_population_size(smoke_cfg: EvoluxConfig, tmp_path: Path) -> None:
    run = assemble_from_config(smoke_cfg, run_dir=tmp_path / "run")
    n = len(run.population.genomes)
    run.close()
    assert n == smoke_cfg.population.size


def test_assemble_step_is_zero(assembled: AssembledRun) -> None:
    assert assembled.step == 0


def test_assemble_device_is_cpu(assembled: AssembledRun) -> None:
    assert assembled.device == torch.device("cpu")


# ── EvolutionLoop ─────────────────────────────────────────────────────────────


def test_loop_run_increments_step(assembled: AssembledRun) -> None:
    loop = EvolutionLoop(assembled)
    loop.run()
    assert assembled.step == assembled.cfg.simulation.max_generations


def test_loop_run_creates_log_entries(assembled: AssembledRun) -> None:
    loop = EvolutionLoop(assembled)
    loop.run()

    log_path = assembled.run_dir / "logs" / "events.jsonl"
    assert log_path.is_file()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 1


def test_loop_log_contains_fitness_scalars(assembled: AssembledRun) -> None:
    loop = EvolutionLoop(assembled)
    loop.run()

    log_path = assembled.run_dir / "logs" / "events.jsonl"
    records = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    keys = {r.get("key") for r in records}
    assert "fitness/mean" in keys
    assert "fitness/max" in keys


def test_loop_max_generations_override(assembled: AssembledRun) -> None:
    loop = EvolutionLoop(assembled)
    loop.run(max_generations=1)
    assert assembled.step == 1


def test_loop_run_is_callable_multiple_times(assembled: AssembledRun) -> None:
    """Calling run() twice accumulates step correctly."""
    loop = EvolutionLoop(assembled)
    loop.run(max_generations=1)
    loop.run(max_generations=1)
    assert assembled.step == 2


def test_loop_population_fitness_updated(assembled: AssembledRun) -> None:
    loop = EvolutionLoop(assembled)
    loop.run(max_generations=1)
    # After step_generation, elites retain their score; offspring get -inf.
    # At least one value should be finite (the elite).
    assert assembled.population.fitness.isfinite().any()


# ── Smoke run timing ──────────────────────────────────────────────────────────


def test_smoke_run_completes_without_error(smoke_cfg: EvoluxConfig, tmp_path: Path) -> None:
    """2 gens x 8 steps x 4 agents on CPU must complete without raising."""
    run = assemble_from_config(smoke_cfg, run_dir=tmp_path / "smoke_run")
    EvolutionLoop(run).run()
    run.close()


# ── Checkpoint ────────────────────────────────────────────────────────────────


def test_checkpoint_save_creates_file(assembled: AssembledRun, tmp_path: Path) -> None:
    EvolutionLoop(assembled).run(max_generations=1)
    ckpt_path = tmp_path / "ckpt.pkl"
    checkpoint.save(assembled, ckpt_path)
    assert ckpt_path.is_file()


def test_checkpoint_load_step(assembled: AssembledRun, tmp_path: Path) -> None:
    EvolutionLoop(assembled).run(max_generations=1)
    ckpt_path = tmp_path / "ckpt.pkl"
    checkpoint.save(assembled, ckpt_path)

    ckpt = checkpoint.load(ckpt_path)
    assert ckpt.step == 1


def test_checkpoint_load_genome_count(assembled: AssembledRun, tmp_path: Path) -> None:
    EvolutionLoop(assembled).run(max_generations=1)
    ckpt_path = tmp_path / "ckpt.pkl"
    checkpoint.save(assembled, ckpt_path)

    ckpt = checkpoint.load(ckpt_path)
    assert len(ckpt.genomes) == len(assembled.population.genomes)


def test_checkpoint_load_fitness_preserved(assembled: AssembledRun, tmp_path: Path) -> None:
    EvolutionLoop(assembled).run(max_generations=1)
    fitness_before = assembled.population.fitness.clone()

    ckpt_path = tmp_path / "ckpt.pkl"
    checkpoint.save(assembled, ckpt_path)

    ckpt = checkpoint.load(ckpt_path)
    assert torch.allclose(fitness_before, ckpt.fitness)


def test_checkpoint_restore_updates_step(assembled: AssembledRun, tmp_path: Path) -> None:
    EvolutionLoop(assembled).run(max_generations=1)
    ckpt_path = tmp_path / "ckpt.pkl"
    checkpoint.save(assembled, ckpt_path)

    assembled.step = 0  # reset manually
    checkpoint.restore(assembled, ckpt_path)
    assert assembled.step == 1


def test_checkpoint_restore_updates_population(assembled: AssembledRun, tmp_path: Path) -> None:
    EvolutionLoop(assembled).run(max_generations=1)
    genomes_saved = [g.params.clone() for g in assembled.population.genomes]  # type: ignore[union-attr]

    ckpt_path = tmp_path / "ckpt.pkl"
    checkpoint.save(assembled, ckpt_path)

    # Replace population with dummy genomes.
    from evolux.genome.direct import DirectGenome

    assembled.population.genomes = [DirectGenome(torch.zeros(1)) for _ in range(4)]

    checkpoint.restore(assembled, ckpt_path)
    genomes_restored = [g.params.clone() for g in assembled.population.genomes]  # type: ignore[union-attr]

    for saved, restored in zip(genomes_saved, genomes_restored, strict=True):
        assert torch.allclose(saved, restored)


def test_checkpoint_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checkpoint.load(tmp_path / "nonexistent.pkl")


def test_checkpoint_bit_exact(smoke_cfg: EvoluxConfig, tmp_path: Path) -> None:
    """Resuming from a checkpoint produces the same evolutionary outcome.

    The stub world returns identical observations for a given (seed, gen) pair
    regardless of the run instance, so the evolutionary trajectory after
    checkpoint restore must match the uninterrupted run exactly.
    """
    # ── run1: run gen 0, save, continue for gen 1 ────────────────────────────
    run1 = assemble_from_config(smoke_cfg, run_dir=tmp_path / "run1")
    loop1 = EvolutionLoop(run1)
    loop1.run(max_generations=1)  # gen 0 → step = 1

    ckpt_path = tmp_path / "ckpt.pkl"
    checkpoint.save(run1, ckpt_path)

    loop1.run(max_generations=1)  # gen 1 → step = 2
    params_run1 = [g.params.clone() for g in run1.population.genomes]  # type: ignore[union-attr]
    run1.close()

    # ── run2: fresh assemble, restore checkpoint, run gen 1 ──────────────────
    run2 = assemble_from_config(smoke_cfg, run_dir=tmp_path / "run2")
    checkpoint.restore(run2, ckpt_path)
    loop2 = EvolutionLoop(run2)
    loop2.run(max_generations=1)  # gen 1 → step = 2
    params_run2 = [g.params.clone() for g in run2.population.genomes]  # type: ignore[union-attr]
    run2.close()

    for p1, p2 in zip(params_run1, params_run2, strict=True):
        assert torch.allclose(p1, p2), "Checkpoint restoration is not bit-exact"


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_validate_smoke() -> None:
    """``evolux validate configs/experiments/smoke.yaml`` exits 0."""
    from typer.testing import CliRunner

    from evolux.orchestrator.cli import app

    config_path = _REPO_ROOT / "configs" / "experiments" / "smoke.yaml"
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(config_path)])
    assert result.exit_code == 0, result.output


def test_cli_run_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``evolux run`` produces runs/<id>/ with config copy + log entries."""
    from typer.testing import CliRunner

    from evolux.orchestrator.cli import app

    config_path = _REPO_ROOT / "configs" / "experiments" / "smoke.yaml"
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "run_dir=" in result.output

    # Check run directory structure.
    run_dirs = sorted((tmp_path / "runs").iterdir())
    assert len(run_dirs) >= 1
    run_dir = run_dirs[0]
    assert (run_dir / "config.yaml").is_file()

    log_path = run_dir / "logs" / "events.jsonl"
    assert log_path.is_file()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 1
