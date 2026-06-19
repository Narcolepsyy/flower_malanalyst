"""Run resource-aware FL experiments with Flower's simulation API."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import flwr as fl
import pandas as pd
from flwr.common import ndarrays_to_parameters

from federated_malware.dataset_utils import create_noniid_partitions, create_partitions, load_malmem
from federated_malware.experiment_utils import (
    build_experiment_metadata,
    clone_partitions_with_label_flip,
    load_experiment_results,
    resolve_resource_config,
    weighted_metrics,
)
from federated_malware.model_utils import (
    TrainConfig,
    build_model,
    make_central_eval_fn,
)
from federated_malware.strategy_factory import create_strategy

LOGGER = logging.getLogger(__name__)


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


def run_experiment(
    agg_method: str,
    num_rounds: int | None = None,
    num_clients: int | None = None,
    model_name: str = "logreg",
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float = 0.01,
    preset: str = "dev",
    partition_method: str = "iid",
    noniid_alpha: float = 0.5,
    seed: int = 42,
    client_num_cpus: float | None = None,
    client_num_gpus: float | None = None,
    max_update_norm: float | None = None,
    server_dp_noise: float = 0.0,
    quantization_bits: int | None = None,
    topk_ratio: float | None = None,
    fedprox_mu: float = 0.0,
    malicious_clients: int = 0,
) -> Dict[str, Any]:
    """Run a single FL experiment with the specified aggregation method."""
    resource_config = resolve_resource_config(
        preset=preset,
        model_name=model_name,
        num_clients=num_clients,
        num_rounds=num_rounds,
        epochs=epochs,
        batch_size=batch_size,
        client_num_cpus=client_num_cpus,
        client_num_gpus=client_num_gpus,
    )
    num_clients = int(resource_config["num_clients"])
    num_rounds = int(resource_config["num_rounds"])
    epochs = int(resource_config["epochs"])
    batch_size = int(resource_config["batch_size"])
    
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("Running experiment: %s / %s / %s", agg_method.upper(), model_name, resource_config['preset'])
    LOGGER.info("=" * 60)
    
    # Load data and create partitions
    x, y, _ = load_malmem("Obfuscated-MalMem2022.csv")
    if partition_method == "noniid":
        partitions, (test_x, test_y) = create_noniid_partitions(
            x, y, num_clients=num_clients, alpha=noniid_alpha, seed=seed
        )
    else:
        partitions, (test_x, test_y) = create_partitions(x, y, num_clients=num_clients, seed=seed)
    partitions = clone_partitions_with_label_flip(partitions, malicious_clients)
    
    n_features = x.shape[1]
    train_cfg = TrainConfig(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        fedprox_mu=fedprox_mu,
    )
    
    # Create client function
    client_fn = create_client_fn(partitions, train_cfg, model_name)
    if agg_method == "catboost" and model_name != "catboost":
        raise ValueError("--methods catboost requires --model catboost")
    
    # Set up metrics file path
    metrics_file = f"state/metrics_{agg_method}.json"
    model_file = f"state/model_{agg_method}.npz"
    
    # Create strategy using the factory (single entry-point)
    base_kwargs = dict(
        log_file=metrics_file,
        model_log_path=model_file,
        reset_metrics=True,
        metadata=build_experiment_metadata(
            entrypoint="run_experiments.py",
            preset=resource_config["preset"],
            agg_method=agg_method,
            model_name=model_name,
            num_clients=num_clients,
            num_rounds=num_rounds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            partition_method=partition_method,
            noniid_alpha=noniid_alpha if partition_method == "noniid" else None,
            seed=seed,
            client_num_cpus=resource_config["client_num_cpus"],
            client_num_gpus=resource_config["client_num_gpus"],
            max_update_norm=max_update_norm,
            server_dp_noise=server_dp_noise,
            quantization_bits=quantization_bits,
            topk_ratio=topk_ratio,
            fedprox_mu=fedprox_mu,
            malicious_clients=malicious_clients,
        ),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_metrics,
        evaluate_fn=make_central_eval_fn(model_name, n_features, test_x, test_y, seed),
    )
    
    # Initialize model parameters
    init_model = build_model(model_name, n_features, seed)
    init_params = ndarrays_to_parameters(init_model.get_parameters())
    base_kwargs["initial_parameters"] = init_params
    
    strategy = create_strategy(
        agg_method=agg_method,
        max_update_norm=max_update_norm,
        server_dp_noise=server_dp_noise,
        quantization_bits=quantization_bits,
        topk_ratio=topk_ratio,
        seed=seed,
        **base_kwargs,
    )
    
    # Run simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": resource_config["client_num_cpus"],
            "num_gpus": resource_config["client_num_gpus"],
        },
    )
    
    # Load and return results
    result = load_experiment_results(metrics_file)
    if result:
        return {"method": agg_method, **result}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="dev", choices=["quick", "dev", "overnight", "quantum-quick"])
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--client-num-cpus", type=float, default=None)
    parser.add_argument("--client-num-gpus", type=float, default=None)
    parser.add_argument("--max-update-norm", type=float, default=None)
    parser.add_argument("--server-dp-noise", type=float, default=0.0)
    parser.add_argument("--quantization-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--topk-ratio", type=float, default=None)
    parser.add_argument("--fedprox-mu", type=float, default=0.0)
    parser.add_argument("--malicious-clients", type=int, default=0)
    parser.add_argument(
        "--model",
        default="logreg",
        choices=["logreg", "mlp", "dp-mlp", "catboost", "hybrid-quantum"],
    )
    parser.add_argument("--partition-method", default="iid", choices=["iid", "noniid"])
    parser.add_argument("--noniid-alpha", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fedavg", "median", "krum"],
        choices=["fedavg", "median", "trimmed", "krum", "bulyan", "mom", "catboost"],
    )
    args = parser.parse_args()
    
    # Run experiments
    all_results = []
    for method in args.methods:
        try:
            result = run_experiment(
                agg_method=method,
                num_rounds=args.num_rounds,
                num_clients=args.num_clients,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                preset=args.preset,
                partition_method=args.partition_method,
                noniid_alpha=args.noniid_alpha,
                seed=args.seed,
                client_num_cpus=args.client_num_cpus,
                client_num_gpus=args.client_num_gpus,
                max_update_norm=args.max_update_norm,
                server_dp_noise=args.server_dp_noise,
                quantization_bits=args.quantization_bits,
                topk_ratio=args.topk_ratio,
                fedprox_mu=args.fedprox_mu,
                malicious_clients=args.malicious_clients,
            )
            all_results.append(result)
        except Exception as e:
            LOGGER.error("Error running %s: %s", method, e)
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
    
    LOGGER.info("Results saved to %s", results_path)
    LOGGER.info("Summary saved to %s", summary_path)
    
    # Create merged metrics for comparison dashboard
    merged = {method: {} for method in args.methods}
    for result in all_results:
        if "error" not in result:
            merged[result["method"]] = result
    
    with open("state/metrics_comparison.json", "w") as f:
        json.dump(merged, f, indent=2)


if __name__ == "__main__":
    main()
