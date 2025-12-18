"""
Run a single FL experiment with specified configuration.
Called by the interactive dashboard.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters

from federated_malware.dataset_utils import (
    create_partitions,
    create_noniid_partitions,
    load_malmem,
)
from federated_malware.model_utils import NumpyLogisticModel, TorchMLPModel, CatBoostModel, HybridQuantumModel, TrainConfig
from federated_malware.strategy import LoggedFedAvg, RobustLoggedFedAvg, CatBoostLoggedFedAvg


def _weighted_metrics(metrics):
    if not metrics:
        return {}
    total_examples = sum(num for num, _ in metrics)
    if total_examples == 0:
        return {}
    keys = set().union(*(m.keys() for _, m in metrics))
    aggregated = {}
    for k in keys:
        aggregated[k] = sum(num * m.get(k, 0.0) for num, m in metrics) / total_examples
    return aggregated


def create_client_fn(partitions, train_cfg, model_name="logreg"):
    """Factory function for creating Flower clients."""
    def client_fn(cid: str):
        from client import MalwareClient
        return MalwareClient(
            cid=int(cid),
            partitions=partitions,
            train_cfg=train_cfg,
            model_name=model_name
        ).to_client()
    return client_fn


def run_single_experiment(
    agg_method: str,
    num_rounds: int,
    num_clients: int,
    model_name: str,
    partition_method: str,
    noniid_alpha: float,
) -> dict:
    """Run a single experiment and return results."""
    
    print(f"Loading dataset...")
    x, y, _ = load_malmem("Obfuscated-MalMem2022.csv")
    
    print(f"Creating {partition_method} partitions for {num_clients} clients...")
    if partition_method == "noniid":
        partitions, _ = create_noniid_partitions(
            x, y, num_clients=num_clients, alpha=noniid_alpha
        )
    else:
        partitions, _ = create_partitions(x, y, num_clients=num_clients)
    
    n_features = x.shape[1]
    train_cfg = TrainConfig(lr=0.05, epochs=2, batch_size=32)
    
    client_fn = create_client_fn(partitions, train_cfg, model_name)
    
    # Set up metrics file
    metrics_file = f"state/metrics_{agg_method}.json"
    model_file = f"state/model_{agg_method}.npz"
    
    base_kwargs = dict(
        log_file=metrics_file,
        model_log_path=model_file,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=_weighted_metrics,
    )
    
    # Initialize model
    if model_name == "logreg":
        init_model = NumpyLogisticModel(n_features=n_features)
    elif model_name == "catboost":
        init_model = CatBoostModel(n_features=n_features)
    elif model_name == "hybrid-quantum":
        init_model = HybridQuantumModel(n_features=n_features)
    else:
        init_model = TorchMLPModel(n_features=n_features, hidden1=128, hidden2=64)
    
    init_params = ndarrays_to_parameters(init_model.get_parameters())
    base_kwargs["initial_parameters"] = init_params
    
    # Choose strategy based on model type
    if model_name == "catboost":
        # CatBoost needs special strategy (no parameter averaging)
        strategy = CatBoostLoggedFedAvg(**base_kwargs)
    elif agg_method == "fedavg":
        strategy = LoggedFedAvg(**base_kwargs)
    else:
        strategy = RobustLoggedFedAvg(
            agg_method=agg_method,
            trim_ratio=0.1,
            krum_f=1,
            flanders_z=None,
            **base_kwargs,
        )
    
    print(f"Starting FL simulation with {agg_method}...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Load results
    metrics_path = Path(metrics_file)
    if metrics_path.exists():
        results = json.loads(metrics_path.read_text())
        return {
            "method": agg_method,
            "rounds": results.get("rounds", []),
            "loss": results.get("loss", []),
            "accuracy": results.get("accuracy", []),
            "precision": results.get("precision", []),
            "recall": results.get("recall", []),
            "f1": results.get("f1", []),
        }
    return {"method": agg_method, "error": "No metrics found"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg-method", default="fedavg", choices=["fedavg", "median", "krum", "trimmed"])
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--model", default="logreg", choices=["logreg", "mlp", "catboost", "hybrid-quantum"])
    parser.add_argument("--partition-method", default="iid", choices=["iid", "noniid"])
    parser.add_argument("--noniid-alpha", type=float, default=0.5)
    args = parser.parse_args()
    
    result = run_single_experiment(
        agg_method=args.agg_method,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        model_name=args.model,
        partition_method=args.partition_method,
        noniid_alpha=args.noniid_alpha,
    )
    
    # Append to experiment results
    results_path = Path("state/experiment_results.json")
    existing = []
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text())
        except:
            pass
    
    # Update or append result for this method
    found = False
    for i, r in enumerate(existing):
        if r.get("method") == result["method"]:
            existing[i] = result
            found = True
            break
    if not found:
        existing.append(result)
    
    results_path.write_text(json.dumps(existing, indent=2))
    print(f"Results saved to {results_path}")
    
    # Print summary
    if "accuracy" in result and result["accuracy"]:
        final_acc = result["accuracy"][-1]
        final_f1 = result["f1"][-1]
        print(f"\nFinal Results for {args.agg_method.upper()}:")
        print(f"  Accuracy: {final_acc*100:.2f}%")
        print(f"  F1 Score: {final_f1*100:.2f}%")


if __name__ == "__main__":
    main()
