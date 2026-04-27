"""Statistics logger: writes generation-level metrics to CSV and JSON."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StatsLogger:
    """Appends per-generation statistics to a CSV file and keeps a JSON snapshot."""

    _FIELDS = [
        "generation",
        "best_fitness",
        "mean_fitness",
        "std_fitness",
        "n_creatures",
        "n_species",
    ]

    def __init__(self, output_dir: Path | str) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._dir / "stats.csv"
        self._json_path = self._dir / "stats.json"
        self._rows: list[dict] = []
        self._csv_file = self._csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._FIELDS, extrasaction="ignore")
        self._writer.writeheader()

    def log(self, stats: dict) -> None:
        self._rows.append(stats)
        self._writer.writerow(stats)
        self._csv_file.flush()

    def save_json(self) -> None:
        with self._json_path.open("w") as f:
            json.dump(self._rows, f, indent=2)

    def close(self) -> None:
        self._csv_file.close()
        self.save_json()
