# viz — SPEC

| Field | Value |
| --- | --- |
| Layer | 6 |
| Phase | 1 (tensorboard) → 4 (web dashboard, video) |
| Depends on | `core`, `tensors` (read-only of any state passed in) |
| Optional dep | `tensorboard`, `wandb`, `matplotlib`, `imageio` (`evolux[viz]`); `fastapi`, `websockets`, three.js (`evolux[web]`) |

## Mission

Make experiments observable: scalar logging, episode rendering, MAP-Elites
heatmaps, lineage trees, and a real-time web dashboard for live runs.

## Submodules

| File | Purpose | Phase |
| --- | --- | --- |
| `loggers.py`     | `TensorboardLogger`, `WandbLogger`, `JsonlLogger` | 1 |
| `renderers.py`   | Top-down agent renderer (PNG/GIF/MP4) | 1 |
| `qd_plots.py`    | MAP-Elites heatmaps, novelty plots | 3 |
| `lineage.py`     | Phylogenetic tree dump (graphviz) | 3 |
| `web/`           | FastAPI + websockets + Three.js dashboard | 4 |

## Acceptance tests

- Loggers handle scalar / hist / image without crashing on missing backend (no-op).
- Renderer produces a non-empty PNG for a trivial trajectory.
- Web dashboard serves index.html and accepts a websocket connection.

## Performance budget

- Logging overhead: ≤ 1% of step time.
- Rendering ≤ 100 ms per episode (offline / on-demand).
