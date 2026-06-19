"""Backwards-compatibility shim.

New code should use ``run_experiments.run_experiment`` directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_experiments import run_experiment


def run_single_experiment(**kwargs):
    """Thin wrapper around run_experiment for backwards compatibility."""
    return run_experiment(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run a single FL experiment.")
    parser.add_argument(
        "--agg-method",
        default="fedavg",
        choices=["fedavg", "median", "krum", "trimmed", "bulyan", "mom", "catboost"],
    )
    parser.add_argument("--preset", default="quick", choices=["quick", "dev", "overnight", "quantum-quick"])
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
    args = parser.parse_args()

    result = run_experiment(
        agg_method=args.agg_method,
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

    # Append / update in experiment_results.json
    results_path = Path("state/experiment_results.json")
    existing: list = []
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            pass

    found = False
    for i, r in enumerate(existing):
        if r.get("method") == result.get("method"):
            existing[i] = result
            found = True
            break
    if not found:
        existing.append(result)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(existing, indent=2))
    print(f"Results saved to {results_path}")

    if "accuracy" in result and result["accuracy"]:
        final_acc = result["accuracy"][-1]
        final_f1 = result["f1"][-1]
        print(f"\nFinal Results for {args.agg_method.upper()}:")
        print(f"  Accuracy: {final_acc * 100:.2f}%")
        print(f"  F1 Score: {final_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()
