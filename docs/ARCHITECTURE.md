# Architecture

The system has four runtime surfaces:

- Flower server with a pluggable strategy.
- N Flower clients, each owning one prepared or on-the-fly dataset partition.
- Experiment runners using `flwr.simulation.start_simulation` for local deterministic runs.
- Canonical dashboard at `dashboard_interactive.py`.

Artifacts:

- `state/metrics*.json`: per-round metrics plus metadata and audit hashes.
- `state/model*.npz`: saved global parameters.
- `state/partitions/<seed>/<scheme>/cid_<id>.npz`: optional prepared client partitions.
- `state/explanations.json`: XAI output for dashboards.

See `docs/architecture.png` for the visual overview.
