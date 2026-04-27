"""sim_env — Simulated Evolution Framework."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sim_env")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
