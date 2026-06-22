# API Reference

Primary public modules:

- `federated_malware.dataset_utils`: dataset loading, train-safe scaling, IID/Non-IID partitions, partition cache helpers.
- `federated_malware.model_utils`: model wrappers implementing `get_parameters`, `set_parameters`, `train_epochs`, and `evaluate`.
- `federated_malware.strategy`: Flower strategies for logging, robust aggregation, CatBoost, clipping, server-DP, compression, and audit hashes.
- `federated_malware.experiment_utils`: resource presets and metadata helpers.
- `federated_malware.strategy_factory`: strategy construction.

To generate expanded HTML docs:

```bash
python -m pip install pdoc
pdoc federated_malware -o docs/api
```
