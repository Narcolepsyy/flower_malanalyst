## Federated Malware Detection (Flower)

This repo contains a minimal Flower simulation that trains a binary malware detector on the Obfuscated-MalMem2022 dataset. It implements:

- Custom Flower strategy that logs aggregated metrics to `state/metrics.json`.
- NumPy-based logistic regression model for lightweight client updates.
- Flask dashboard that polls the metrics file for near real-time monitoring.

### Quickstart

1) Install dependencies (recommend using the provided `.venv`):
```
. .venv/bin/activate
pip install -r requirements.txt
```

2) Start the Flower server (writes metrics to `state/metrics.json`). You can choose robust aggregation:
```
python server.py --rounds 3 --address 0.0.0.0:8080 \
  --agg-method median          # options: fedavg (default), median, trimmed, krum
# Optional knobs:
# --trim-ratio 0.1             # for trimmed mean
# --krum-f 1                   # assumed Byzantine count for krum scoring
# --flanders-z 3.0             # drop updates with norm z-score above threshold (simple FLANDERS-like filter)
```

3) Launch a few clients in separate terminals (adjust `--cid` and `--num-clients` so they match). Choose model via `--model logreg|mlp`:
```
python client.py --cid 0 --num-clients 2 --epochs 1 --batch-size 64 --lr 0.01 --model mlp --server-address 0.0.0.0:8080
python client.py --cid 1 --num-clients 2 --epochs 1 --batch-size 64 --lr 0.01 --model mlp --server-address 0.0.0.0:8080
```
Clients automatically read the MalMem dataset from `Obfuscated-MalMem2022.csv`, standardize features, and receive a stratified partition.

4) Open the live dashboard (Flask):
```
python dashboard_flask.py --host 0.0.0.0 --port 8501 --metrics state/metrics.json
```
Visit http://localhost:8501 to see loss/accuracy/F1 charts; it polls `state/metrics.json` every 2 seconds.

### Explainability (XAI)

- The server now persists the aggregated global model to `state/latest_model.npz` (configurable with `--model-save`).
- After training, generate feature attributions and feed them to the dashboard:
  ```
  python explain.py \
    --data-path Obfuscated-MalMem2022.csv \
    --model-weights state/latest_model.npz \
    --top-k 12 \
    --background-size 256 \
    --method auto \
    --output state/explanations.json
  ```
- The Flask dashboard automatically reads `state/explanations.json` (polls every 5 seconds) and shows the top features and model type. For logistic regression, importance is the absolute coefficient; for the MLP, it uses gradient-based saliency averaged over a background sample.

### Project layout

- `federated_malware/dataset_utils.py`: Loading, scaling, and stratified partitioning of MalMem.
- `federated_malware/model_utils.py`: Lightweight NumPy logistic regression with manual SGD.
- `federated_malware/strategy.py`: `LoggedFedAvg` plus `RobustLoggedFedAvg` (median/trimmed/Krum + FLANDERS-like filter) for aggregation and logging.
- `server.py`: Flower server entrypoint with weighted aggregation of accuracy/precision/recall/F1.
- `client.py`: Flower NumPyClient implementation returning richer metrics; supports `logreg` (NumPy) and `mlp` (PyTorch).
- `dashboard_flask.py`: Flask monitoring UI (polls `state/metrics.json` and renders Plotly charts).
- `state/metrics.json`: Created/updated by the server after each evaluation aggregation.
