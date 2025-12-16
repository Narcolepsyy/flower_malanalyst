"""
Run FL experiments with different aggregation strategies and collect results.
This script runs server and clients programmatically using Flower's simulation API.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import ndarrays_to_parameters

from federated_malware.dataset_utils import create_partitions, load_malmem
from federated_malware.model_utils import NumpyLogisticModel, TorchMLPModel, TrainConfig
from federated_malware.strategy import LoggedFedAvg, RobustLoggedFedAvg


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


def _weighted_metrics(metrics):
    """Aggregate client metrics weighted by num_examples."""
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


def run_experiment(
    agg_method: str,
    num_rounds: int = 5,
    num_clients: int = 2,
    model_name: str = "logreg",
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.01,
) -> Dict[str, Any]:
    """Run a single FL experiment with the specified aggregation method."""
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {agg_method.upper()}")
    print(f"{'='*60}")
    
    # Load data and create partitions
    x, y, _ = load_malmem("Obfuscated-MalMem2022.csv")
    partitions, (test_x, test_y) = create_partitions(x, y, num_clients=num_clients)
    
    n_features = x.shape[1]
    train_cfg = TrainConfig(lr=lr, epochs=epochs, batch_size=batch_size)
    
    # Create client function
    client_fn = create_client_fn(partitions, train_cfg, model_name)
    
    # Set up metrics file path
    metrics_file = f"state/metrics_{agg_method}.json"
    model_file = f"state/model_{agg_method}.npz"
    
    # Create strategy based on aggregation method
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
    
    # Initialize model parameters
    if model_name == "logreg":
        init_model = NumpyLogisticModel(n_features=n_features)
    else:
        init_model = TorchMLPModel(n_features=n_features, hidden1=128, hidden2=64)
    
    init_params = ndarrays_to_parameters(init_model.get_parameters())
    base_kwargs["initial_parameters"] = init_params
    
    if agg_method == "fedavg":
        strategy = LoggedFedAvg(**base_kwargs)
    else:
        strategy = RobustLoggedFedAvg(
            agg_method=agg_method,
            trim_ratio=0.1,
            krum_f=1,
            flanders_z=None,
            **base_kwargs,
        )
    
    # Run simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Load and return results
    metrics_path = Path(metrics_file)
    if metrics_path.exists():
        with open(metrics_path) as f:
            results = json.load(f)
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


def summarize_results(all_results: List[Dict]) -> pd.DataFrame:
    """Create summary DataFrame from all experiment results."""
    summary = []
    for result in all_results:
        if "error" in result:
            continue
        
        method = result["method"]
        # Get final metrics (last round)
        final_acc = result["accuracy"][-1] if result["accuracy"] else 0
        final_loss = result["loss"][-1] if result["loss"] else 0
        final_precision = result["precision"][-1] if result["precision"] else 0
        final_recall = result["recall"][-1] if result["recall"] else 0
        final_f1 = result["f1"][-1] if result["f1"] else 0
        
        # Get best metrics
        best_acc = max(result["accuracy"]) if result["accuracy"] else 0
        best_f1 = max(result["f1"]) if result["f1"] else 0
        
        summary.append({
            "Method": method.upper(),
            "Final Accuracy": f"{final_acc:.4f}",
            "Final F1": f"{final_f1:.4f}",
            "Final Precision": f"{final_precision:.4f}",
            "Final Recall": f"{final_recall:.4f}",
            "Final Loss": f"{final_loss:.4f}",
            "Best Accuracy": f"{best_acc:.4f}",
            "Best F1": f"{best_f1:.4f}",
        })
    
    return pd.DataFrame(summary)


def main():
    # Configuration
    NUM_ROUNDS = 5
    NUM_CLIENTS = 2
    MODEL_NAME = "logreg"  # Use logreg for faster experiments
    
    # Methods to compare
    methods = ["fedavg", "median", "krum"]
    
    # Run experiments
    all_results = []
    for method in methods:
        try:
            result = run_experiment(
                agg_method=method,
                num_rounds=NUM_ROUNDS,
                num_clients=NUM_CLIENTS,
                model_name=MODEL_NAME,
                epochs=2,
                batch_size=32,
                lr=0.05,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error running {method}: {e}")
            all_results.append({"method": method, "error": str(e)})
    
    # Create summary
    summary_df = summarize_results(all_results)
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save results
    results_path = Path("state/experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    summary_path = Path("state/results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nResults saved to {results_path}")
    print(f"Summary saved to {summary_path}")
    
    # Create merged metrics for comparison dashboard
    merged = {"fedavg": {}, "median": {}, "krum": {}}
    for result in all_results:
        if "error" not in result:
            merged[result["method"]] = result
    
    with open("state/metrics_comparison.json", "w") as f:
        json.dump(merged, f, indent=2)


if __name__ == "__main__":
    main()
