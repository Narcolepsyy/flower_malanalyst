"""
Prepare per-client dataset partitions for standalone Flower clients.
"""

from __future__ import annotations

import argparse

from federated_malware.dataset_utils import (
    create_noniid_partitions,
    create_partitions,
    load_malmem,
    partition_cache_dir,
    save_partition_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached MalMem partitions")
    parser.add_argument("--data-path", default="Obfuscated-MalMem2022.csv")
    parser.add_argument("--output-root", default="state/partitions")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition-method", choices=["iid", "noniid"], default="iid")
    parser.add_argument("--noniid-alpha", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y, _ = load_malmem(args.data_path)
    if args.partition_method == "noniid":
        partitions, test_set = create_noniid_partitions(
            x,
            y,
            num_clients=args.num_clients,
            alpha=args.noniid_alpha,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        partitions, test_set = create_partitions(
            x,
            y,
            num_clients=args.num_clients,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    output_dir = partition_cache_dir(
        seed=args.seed,
        partition_method=args.partition_method,
        root=args.output_root,
    )
    save_partition_cache(
        partitions,
        test_set,
        output_dir,
        metadata={
            "data_path": args.data_path,
            "seed": args.seed,
            "partition_method": args.partition_method,
            "noniid_alpha": args.noniid_alpha if args.partition_method == "noniid" else None,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        overwrite=args.force,
    )
    print(f"Saved {len(partitions)} client partitions to {output_dir}")


if __name__ == "__main__":
    main()
