"""
Flower server entrypoint using the custom LoggedFedAvg strategy.
"""

from __future__ import annotations

import argparse
import logging

import flwr as fl

from federated_malware.experiment_utils import build_experiment_metadata, weighted_metrics
from federated_malware.logging_utils import configure_logging
from federated_malware.strategy_factory import create_strategy

LOGGER = logging.getLogger(__name__)


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
        choices=["fedavg", "median", "trimmed", "krum", "bulyan", "mom", "catboost"],
        help="Aggregation rule for model updates",
    )
    parser.add_argument(
        "--reset-metrics",
        action="store_true",
        help="Overwrite the metrics log at startup instead of appending to an existing file",
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
    parser.add_argument("--mom-groups", type=int, default=3, help="Groups for median-of-means")
    parser.add_argument("--max-update-norm", type=float, default=None, help="Clip client update norm")
    parser.add_argument("--server-dp-noise", type=float, default=0.0, help="Gaussian server-DP noise multiplier")
    parser.add_argument("--quantization-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--topk-ratio", type=float, default=None, help="Keep this fraction of update entries")
    parser.add_argument("--seed", type=int, default=42, help="Server RNG seed")
    # SSL/TLS options for secure communication (mTLS)
    parser.add_argument("--ssl-certfile", type=str, default=None, help="Path to server SSL certificate")
    parser.add_argument("--ssl-keyfile", type=str, default=None, help="Path to server SSL private key")
    parser.add_argument("--ssl-ca-certfile", type=str, default=None, help="Path to CA certificate for client verification")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    base_kwargs = dict(
        log_file=args.log_file,
        model_log_path=args.model_save,
        reset_metrics=args.reset_metrics,
        metadata=build_experiment_metadata(
            entrypoint="server.py",
            agg_method=args.agg_method,
            num_rounds=args.rounds,
            min_clients=args.min_clients,
            seed=args.seed,
            max_update_norm=args.max_update_norm,
            server_dp_noise=args.server_dp_noise,
            quantization_bits=args.quantization_bits,
            topk_ratio=args.topk_ratio,
        ),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        evaluate_metrics_aggregation_fn=weighted_metrics,
    )

    strategy = create_strategy(
        agg_method=args.agg_method,
        trim_ratio=args.trim_ratio,
        krum_f=args.krum_f,
        flanders_z=args.flanders_z,
        mom_groups=args.mom_groups,
        max_update_norm=args.max_update_norm,
        server_dp_noise=args.server_dp_noise,
        quantization_bits=args.quantization_bits,
        topk_ratio=args.topk_ratio,
        seed=args.seed,
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
        LOGGER.info("mTLS enabled with certificate: %s", args.ssl_certfile)

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        certificates=certificates,
    )


if __name__ == "__main__":
    main()
