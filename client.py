"""
Flower NumPyClient for federated malware detection on the MalMem dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import flwr as fl
import numpy as np

from federated_malware.dataset_utils import (
    DatasetPartition,
    create_noniid_partitions,
    create_partitions,
    get_partition_stats,
    load_malmem,
    load_partition_cache,
    partition_cache_dir,
)
from federated_malware.logging_utils import configure_logging
from federated_malware.model_utils import (
    CatBoostModel,
    DPTorchMLPModel,
    HybridQuantumModel,
    NumpyLogisticModel,
    TorchMLPModel,
    TrainConfig,
)

LOGGER = logging.getLogger(__name__)


def _build_logreg(n_features: int, train_cfg: TrainConfig, **kwargs):
    return NumpyLogisticModel(n_features=n_features, lr=train_cfg.lr, seed=train_cfg.seed)


def _build_mlp(n_features: int, train_cfg: TrainConfig, **kwargs):
    return TorchMLPModel(
        n_features=n_features,
        lr=train_cfg.lr,
        hidden1=train_cfg.hidden1,
        hidden2=train_cfg.hidden2,
        seed=train_cfg.seed,
    )


def _build_dp_mlp(n_features: int, train_cfg: TrainConfig, **kwargs):
    return DPTorchMLPModel(
        n_features=n_features,
        lr=train_cfg.lr,
        hidden1=train_cfg.hidden1,
        hidden2=train_cfg.hidden2,
        target_epsilon=kwargs["dp_epsilon"],
        target_delta=kwargs["dp_delta"],
        noise_multiplier=kwargs["dp_noise_multiplier"],
        max_grad_norm=kwargs["dp_max_grad_norm"],
        seed=train_cfg.seed,
    )


def _build_catboost(n_features: int, train_cfg: TrainConfig, **kwargs):
    return CatBoostModel(n_features=n_features, seed=train_cfg.seed)


def _build_hybrid_quantum(n_features: int, train_cfg: TrainConfig, **kwargs):
    return HybridQuantumModel(n_features=n_features, lr=train_cfg.lr, seed=train_cfg.seed)


MODEL_REGISTRY = {
    "logreg": _build_logreg,
    "mlp": _build_mlp,
    "dp-mlp": _build_dp_mlp,
    "catboost": _build_catboost,
    "hybrid-quantum": _build_hybrid_quantum,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated Malware Client")
    parser.add_argument("--cid", type=int, default=0, help="Client ID (0-indexed)")
    parser.add_argument("--num-clients", type=int, default=2, help="Total simulated clients")
    parser.add_argument("--seed", type=int, default=42, help="Partition seed")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden1", type=int, default=128, help="Hidden size 1 (MLP)")
    parser.add_argument("--hidden2", type=int, default=64, help="Hidden size 2 (MLP)")
    parser.add_argument("--fedprox-mu", type=float, default=0.0, help="FedProx proximal strength")
    parser.add_argument(
        "--data-path",
        type=str,
        default="Obfuscated-MalMem2022.csv",
        help="CSV path to the MalMem dataset",
    )
    parser.add_argument(
        "--partition-root",
        type=str,
        default="state/partitions",
        help="Root directory for prepared partitions",
    )
    parser.add_argument(
        "--partition-dir",
        type=str,
        default=None,
        help="Specific directory containing cid_<id>.npz prepared partitions",
    )
    parser.add_argument(
        "--require-prepared-partition",
        action="store_true",
        help="Fail instead of falling back to CSV loading when the prepared partition is missing",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8080",
        help="Flower server address host:port",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "mlp", "dp-mlp", "catboost", "hybrid-quantum"],
        help="Model type: 'logreg', 'mlp', 'dp-mlp', 'catboost', or 'hybrid-quantum'",
    )
    parser.add_argument("--dp-epsilon", type=float, default=1.0, help="Target epsilon for DP-MLP")
    parser.add_argument("--dp-delta", type=float, default=1e-5, help="Target delta for DP-MLP")
    parser.add_argument(
        "--dp-noise-multiplier",
        type=float,
        default=1.0,
        help="Noise multiplier for DP-MLP",
    )
    parser.add_argument(
        "--dp-max-grad-norm",
        type=float,
        default=1.0,
        help="Per-sample gradient clipping norm for DP-MLP",
    )
    # Non-IID data distribution options
    parser.add_argument(
        "--partition-method",
        type=str,
        default="iid",
        choices=["iid", "noniid"],
        help="Data partition method: 'iid' (balanced) or 'noniid' (Dirichlet-based)",
    )
    parser.add_argument(
        "--noniid-alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha for Non-IID partitioning (lower=more heterogeneous)",
    )
    # SSL/TLS options for secure communication
    parser.add_argument("--ssl-ca-certfile", type=str, default=None, help="Path to CA certificate")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


class MalwareClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        partitions: Dict[int, "DatasetPartition"],
        train_cfg: TrainConfig,
        model_name: str = "logreg",
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        dp_noise_multiplier: float = 1.0,
        dp_max_grad_norm: float = 1.0,
    ):
        self.cid = cid
        if cid not in partitions:
            raise ValueError(f"Client id {cid} not in partition map")
        part = partitions[cid]
        self.train_x = part.train_x
        self.train_y = part.train_y
        self.val_x = part.val_x
        self.val_y = part.val_y

        n_features = self.train_x.shape[1]
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'")
        self.model = MODEL_REGISTRY[model_name](
            n_features,
            train_cfg,
            dp_epsilon=dp_epsilon,
            dp_delta=dp_delta,
            dp_noise_multiplier=dp_noise_multiplier,
            dp_max_grad_norm=dp_max_grad_norm,
        )
        self.train_cfg = train_cfg

    def get_parameters(self, config=None) -> List[np.ndarray]:
        return self.model.get_parameters()

    def fit(self, parameters, config=None):
        self.model.set_parameters(parameters)
        self.model.train_epochs(self.train_x, self.train_y, self.train_cfg)
        metrics = self.model.evaluate(self.val_x, self.val_y)
        return self.model.get_parameters(), len(self.train_x), metrics

    def evaluate(self, parameters, config=None):
        self.model.set_parameters(parameters)
        metrics = self.model.evaluate(self.val_x, self.val_y)
        return metrics["loss"], len(self.val_x), {
            key: value for key, value in metrics.items() if key != "loss"
        }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    cid_env = os.getenv("CLIENT_ID")
    cid = int(cid_env) if cid_env is not None else args.cid

    prepared_dir = (
        Path(args.partition_dir)
        if args.partition_dir
        else partition_cache_dir(
            seed=args.seed,
            partition_method=args.partition_method,
            root=args.partition_root,
        )
    )
    prepared_file = prepared_dir / f"cid_{cid}.npz"
    if prepared_file.exists():
        partition = load_partition_cache(prepared_dir, cid)
        partitions = {cid: partition}
        LOGGER.info("Client %s loaded prepared partition from %s", cid, prepared_file)
    else:
        if args.require_prepared_partition:
            raise FileNotFoundError(
                f"Prepared partition {prepared_file} is missing. "
                "Run prepare_partitions.py first or omit --require-prepared-partition."
            )

        x, y, _ = load_malmem(args.data_path)

        # Choose partition method
        if args.partition_method == "noniid":
            partitions, _ = create_noniid_partitions(
                x,
                y,
                num_clients=args.num_clients,
                alpha=args.noniid_alpha,
                seed=args.seed,
            )
            # Print partition statistics for Non-IID analysis
            stats = get_partition_stats(partitions)
            LOGGER.info("Client %s Non-IID partition stats (alpha=%s)", cid, args.noniid_alpha)
            for c, s in stats.items():
                LOGGER.info(
                    "Client %s: %s samples, malware_ratio=%.2f%%",
                    c,
                    s["total"],
                    s["malware_ratio"] * 100,
                )
        else:
            partitions, _ = create_partitions(
                x,
                y,
                num_clients=args.num_clients,
                seed=args.seed,
            )

    train_cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        seed=args.seed,
        fedprox_mu=args.fedprox_mu,
    )
    client = MalwareClient(
        cid,
        partitions,
        train_cfg,
        model_name=args.model,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_max_grad_norm=args.dp_max_grad_norm,
    )

    # SSL/TLS configuration for secure communication
    root_certificates = None
    if args.ssl_ca_certfile:
        with open(args.ssl_ca_certfile, "rb") as f:
            root_certificates = f.read()
    
    # start_client is the preferred API; NumPyClient exposes .to_client()
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
        root_certificates=root_certificates,
    )


if __name__ == "__main__":
    main()
