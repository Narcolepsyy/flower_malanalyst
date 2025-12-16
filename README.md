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

### Non-IID Data Distribution

Demonstrate FL challenges with heterogeneous data using Dirichlet-based partitioning:

```bash
# Low alpha = highly skewed data per client (more challenging)
python client.py --cid 0 --num-clients 2 --partition-method noniid --noniid-alpha 0.1

# High alpha = nearly IID (easier convergence)
python client.py --cid 0 --num-clients 2 --partition-method noniid --noniid-alpha 10.0
```

Alpha values:
- `0.1`: Highly heterogeneous (some clients may have mostly malware or mostly benign)
- `0.5`: Moderate heterogeneity (default)
- `1.0`: Mild heterogeneity
- `10.0`: Nearly IID

### Differential Privacy (Opacus)

Train with formal differential privacy guarantees using Opacus:

```bash
# Install opacus
pip install opacus>=1.4.0

# Use dp-mlp model (DP only works with PyTorch MLP, not logreg)
python client.py --cid 0 --model dp-mlp --dp-epsilon 1.0 --dp-delta 1e-5
```

Parameters:
- `--dp-epsilon`: Privacy budget (lower = more private, default: 1.0)
- `--dp-delta`: Probability of privacy breach (default: 1e-5)
- `--dp-noise-multiplier`: Noise scale (default: 1.0)
- `--dp-max-grad-norm`: Gradient clipping norm (default: 1.0)

### Secure Communication (mTLS)

Enable mutual TLS authentication between server and clients:

```bash
# 1. Generate certificates
chmod +x certs/generate_certs.sh
./certs/generate_certs.sh 2  # Generate for 2 clients

# 2. Start server with SSL
python server.py --rounds 3 --address 0.0.0.0:8080 \
  --ssl-certfile certs/server.crt \
  --ssl-keyfile certs/server.key \
  --ssl-ca-certfile certs/ca.crt

# 3. Start clients with SSL
python client.py --cid 0 --ssl-ca-certfile certs/ca.crt
```

### Experiment Comparison

Run experiments comparing aggregation methods:
```bash
python run_experiments.py  # Compares FedAvg, Median, Krum
python dashboard_comparison.py --port 8502  # View results at localhost:8502
```
