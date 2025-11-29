"""
Flower NumPyClient for federated malware detection on the MalMem dataset.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import flwr as fl
import numpy as np

from federated_malware.dataset_utils import DatasetPartition, create_partitions, load_malmem
from federated_malware.model_utils import NumpyLogisticModel, TorchMLPModel, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated Malware Client")
    parser.add_argument("--cid", type=int, default=0, help="Client ID (0-indexed)")
    parser.add_argument("--num-clients", type=int, default=2, help="Total simulated clients")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden1", type=int, default=128, help="Hidden size 1 (MLP)")
    parser.add_argument("--hidden2", type=int, default=64, help="Hidden size 2 (MLP)")
    parser.add_argument(
        "--data-path",
        type=str,
        default="Obfuscated-MalMem2022.csv",
        help="CSV path to the MalMem dataset",
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
        choices=["logreg", "mlp"],
        help="Model type: 'logreg' (NumPy logistic) or 'mlp' (PyTorch MLP)",
    )
    return parser.parse_args()


class MalwareClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        partitions: Dict[int, "DatasetPartition"],
        train_cfg: TrainConfig,
        model_name: str = "logreg",
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
        if model_name == "logreg":
            self.model = NumpyLogisticModel(n_features=n_features, lr=train_cfg.lr)
        elif model_name == "mlp":
            self.model = TorchMLPModel(
                n_features=n_features,
                lr=train_cfg.lr,
                hidden1=train_cfg.hidden1,
                hidden2=train_cfg.hidden2,
            )
        else:
            raise ValueError(f"Unknown model '{model_name}'")
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
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }


def main() -> None:
    args = parse_args()

    cid_env = os.getenv("CLIENT_ID")
    cid = int(cid_env) if cid_env is not None else args.cid

    x, y, _ = load_malmem(args.data_path)
    partitions, _ = create_partitions(x, y, num_clients=args.num_clients)

    train_cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
    )
    client = MalwareClient(cid, partitions, train_cfg, model_name=args.model)

    # start_client is the preferred API; NumPyClient exposes .to_client()
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
