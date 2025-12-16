"""
Flower server entrypoint using the custom LoggedFedAvg strategy.
"""

from __future__ import annotations

import argparse

import flwr as fl

from federated_malware.strategy import LoggedFedAvg, RobustLoggedFedAvg


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated Malware Server")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument(
        "--min-clients",
        type=int,
        default=1,
        help="Minimum available/eval/fit clients required per round",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="state/metrics.json",
        help="Where to persist aggregated metrics",
    )
    parser.add_argument(
        "--model-save",
        type=str,
        default="state/latest_model.npz",
        help="Where to persist aggregated global model parameters for XAI",
    )
    parser.add_argument(
        "--agg-method",
        type=str,
        default="fedavg",
        choices=["fedavg", "median", "trimmed", "krum"],
        help="Aggregation rule for model updates",
    )
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=0.1,
        help="Trim ratio for trimmed mean (fraction to drop each side)",
    )
    parser.add_argument(
        "--krum-f",
        type=int,
        default=1,
        help="Assumed number of Byzantine clients for Krum scoring",
    )
    parser.add_argument(
        "--flanders-z",
        type=float,
        default=None,
        help="Z-score threshold for FLANDERS-like norm filter; disable if None",
    )
    # SSL/TLS options for secure communication (mTLS)
    parser.add_argument("--ssl-certfile", type=str, default=None, help="Path to server SSL certificate")
    parser.add_argument("--ssl-keyfile", type=str, default=None, help="Path to server SSL private key")
    parser.add_argument("--ssl-ca-certfile", type=str, default=None, help="Path to CA certificate for client verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_kwargs = dict(
        log_file=args.log_file,
        model_log_path=args.model_save,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        evaluate_metrics_aggregation_fn=_weighted_metrics,
    )

    if args.agg_method == "fedavg":
        strategy = LoggedFedAvg(**base_kwargs)
    else:
        strategy = RobustLoggedFedAvg(
            agg_method=args.agg_method,
            trim_ratio=args.trim_ratio,
            krum_f=args.krum_f,
            flanders_z=args.flanders_z,
            **base_kwargs,
        )

    # Load SSL certificates if provided
    certificates = None
    if args.ssl_certfile and args.ssl_keyfile:
        with open(args.ssl_certfile, "rb") as f:
            server_cert = f.read()
        with open(args.ssl_keyfile, "rb") as f:
            server_key = f.read()
        ca_cert = None
        if args.ssl_ca_certfile:
            with open(args.ssl_ca_certfile, "rb") as f:
                ca_cert = f.read()
        certificates = (server_cert, server_key, ca_cert) if ca_cert else (server_cert, server_key)
        print(f"[Server] mTLS enabled with certificate: {args.ssl_certfile}")

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        certificates=certificates,
    )


if __name__ == "__main__":
    main()
