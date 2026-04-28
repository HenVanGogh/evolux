"""Typer CLI entry point — wired to ``evolux = evolux.orchestrator.cli:app`` console script."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="evolux — GPU-native modular evolutionary creature simulator.")


@app.command()
def validate(config: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Validate a YAML config against the EvoluxConfig schema."""
    from evolux.core.config import load_config

    cfg = load_config(config)
    typer.echo(f"OK: {config}\n  device={cfg.simulation.device}  seed={cfg.simulation.seed}")


@app.command()
def run(config: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Run an experiment from a YAML config."""
    from evolux.core.config import load_config
    from evolux.orchestrator.assemble import assemble_from_config
    from evolux.orchestrator.loop import EvolutionLoop

    cfg = load_config(config)
    assembled = assemble_from_config(cfg)
    typer.echo(f"run_dir={assembled.run_dir}")

    loop = EvolutionLoop(assembled)
    loop.run()
    assembled.logger.close()

    typer.echo(f"Done. run_dir={assembled.run_dir}")


@app.command()
def info() -> None:
    """Print version + device info."""
    import torch

    from evolux import __version__

    typer.echo(f"evolux {__version__}")
    typer.echo(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")


if __name__ == "__main__":
    app()
