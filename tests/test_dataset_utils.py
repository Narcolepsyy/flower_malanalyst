import tempfile
import unittest

import numpy as np

from federated_malware.dataset_utils import (
    create_noniid_partitions,
    create_partitions,
    load_partition_cache,
    load_partition_manifest,
    partition_cache_dir,
    save_partition_cache,
)


class DatasetUtilsTest(unittest.TestCase):
    def test_iid_partitions_scale_after_train_test_split(self):
        x = np.arange(2000, dtype=float).reshape(100, 20)
        y = np.array([0, 1] * 50)

        partitions, (x_test, y_test) = create_partitions(x, y, num_clients=5, seed=7)
        train_parts = [part.train_x for part in partitions.values()]
        val_parts = [part.val_x for part in partitions.values()]
        scaled_train = np.vstack(train_parts + val_parts)

        self.assertEqual(len(partitions), 5)
        self.assertEqual(x_test.shape, (10, 20))
        self.assertEqual(y_test.shape, (10,))
        self.assertAlmostEqual(float(scaled_train.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(scaled_train.std()), 1.0, places=10)

    def test_noniid_partitions_create_skewed_client_ratios(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(200, 8))
        y = np.array([0] * 100 + [1] * 100)

        partitions, _ = create_noniid_partitions(x, y, num_clients=4, alpha=0.1, seed=3)
        ratios = [
            float((part.train_y.sum() + part.val_y.sum()) / (len(part.train_y) + len(part.val_y)))
            for part in partitions.values()
        ]

        self.assertGreaterEqual(len(partitions), 2)
        self.assertGreater(max(ratios) - min(ratios), 0.25)

    def test_partition_cache_round_trip_loads_single_client(self):
        x = np.arange(400, dtype=float).reshape(50, 8)
        y = np.array([0, 1] * 25)
        partitions, test_set = create_partitions(x, y, num_clients=2, seed=9)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = partition_cache_dir(seed=9, partition_method="iid", root=tmpdir)
            save_partition_cache(
                partitions,
                test_set,
                output_dir,
                metadata={"seed": 9, "partition_method": "iid"},
            )
            loaded = load_partition_cache(output_dir, cid=1)
            manifest = load_partition_manifest(output_dir)

        self.assertEqual(manifest["seed"], 9)
        self.assertEqual(manifest["partition_method"], "iid")
        self.assertTrue(np.array_equal(loaded.train_x, partitions[1].train_x))
        self.assertTrue(np.array_equal(loaded.train_y, partitions[1].train_y))
        self.assertTrue(np.array_equal(loaded.val_x, partitions[1].val_x))
        self.assertTrue(np.array_equal(loaded.val_y, partitions[1].val_y))


if __name__ == "__main__":
    unittest.main()
