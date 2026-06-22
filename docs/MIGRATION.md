# Migration Notes

## Scaler Leakage Fix

Older runs fit `StandardScaler` over the full dataset in `load_malmem`, before train/test splitting. Current runs load raw features and fit the scaler only after the held-out test split is created.

Expected effect:

- Reported validation/test metrics may change.
- Older `state/*.json` files generated before this change are not directly comparable to new runs.
- New metrics include metadata fields such as seed, partition method, aggregation method, and model type to make comparisons auditable.

## CatBoost Serialization Fix

CatBoost models are now serialized with CatBoost's native model format. Older `.npz` artifacts containing pickled CatBoost bytes should be discarded and regenerated.
