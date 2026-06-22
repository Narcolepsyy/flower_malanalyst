import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from federated_malware.dataset_utils import DatasetPartition
from federated_malware.experiment_utils import (
    clone_partitions_with_label_flip,
    resolve_resource_config,
)
from federated_malware.model_utils import CatBoostModel
from federated_malware.strategy import LoggedFedAvg, RobustLoggedFedAvg
from federated_malware.strategy.aggregators import (
    agg_bulyan,
    agg_krum,
    agg_median,
    agg_median_of_means,
    apply_quantization,
    apply_topk,
    clip_update,
)


class StrategyAndModelTest(unittest.TestCase):
    def test_logged_fedavg_preserves_existing_metrics_without_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            original = {"rounds": [1], "loss": [0.3], "accuracy": [0.9]}
            path.write_text(json.dumps(original))

            LoggedFedAvg(log_file=str(path), model_log_path=None)

            self.assertEqual(json.loads(path.read_text()), original)

    def test_logged_fedavg_reset_writes_metadata_skeleton(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            path.write_text(json.dumps({"rounds": [1]}))

            LoggedFedAvg(
                log_file=str(path),
                model_log_path=None,
                reset_metrics=True,
                metadata={"preset": "quick", "model_name": "logreg"},
            )
            current = json.loads(path.read_text())

            self.assertEqual(current["metadata"]["preset"], "quick")
            self.assertEqual(current["rounds"], [])
            self.assertEqual(current["f1"], [])

    def test_robust_aggregators_match_expected_values(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        strategy = RobustLoggedFedAvg(
            log_file=str(Path(tmpdir.name) / "metrics.json"),
            model_log_path=None,
            reset_metrics=True,
            agg_method="median",
        )
        params = [
            [np.array([1.0, 2.0]), np.array([0.0])],
            [np.array([2.0, 3.0]), np.array([1.0])],
            [np.array([100.0, 200.0]), np.array([50.0])],
        ]

        median = agg_median(params)
        self.assertTrue(np.allclose(median[0], np.array([2.0, 3.0])))
        self.assertTrue(np.allclose(median[1], np.array([1.0])))

        krum = agg_krum(params, f=1)
        self.assertTrue(np.allclose(krum[0], np.array([1.0, 2.0])))

    def test_bulyan_and_median_of_means_drop_obvious_outlier(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        strategy = RobustLoggedFedAvg(
            log_file=str(Path(tmpdir.name) / "metrics.json"),
            model_log_path=None,
            reset_metrics=True,
            agg_method="bulyan",
            krum_f=1,
            mom_groups=2,
        )
        params = [
            [np.array([1.0, 1.0])],
            [np.array([1.1, 1.0])],
            [np.array([0.9, 1.2])],
            [np.array([1.0, 0.8])],
            [np.array([100.0, 100.0])],
        ]

        bulyan = agg_bulyan(params, f=1)
        mom = agg_median_of_means(params[:4], mom_groups=2)

        self.assertLess(float(np.linalg.norm(bulyan[0] - np.array([1.0, 1.0]))), 0.2)
        self.assertTrue(np.allclose(mom[0], np.array([1.0, 1.0])))

    def test_update_transform_helpers(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        strategy = RobustLoggedFedAvg(
            log_file=str(Path(tmpdir.name) / "metrics.json"),
            model_log_path=None,
            reset_metrics=True,
            max_update_norm=1.0,
            quantization_bits=8,
            topk_ratio=0.5,
        )
        update = [np.array([3.0, 4.0, 0.1, 0.0])]

        clipped = clip_update(update, max_update_norm=1.0)
        sparse = apply_topk(update, topk_ratio=0.5)
        quantized = apply_quantization(update, quantization_bits=8)

        self.assertLessEqual(float(np.linalg.norm(clipped[0])), 1.0 + 1e-9)
        self.assertEqual(int(np.count_nonzero(sparse[0])), 2)
        self.assertEqual(quantized[0].shape, update[0].shape)

    def test_metrics_audit_hash_chain_is_appended(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            strategy = LoggedFedAvg(log_file=str(path), model_log_path=None, reset_metrics=True)
            strategy._append_metrics(1, 0.5, {"accuracy": 0.7})
            strategy._append_metrics(2, 0.4, {"accuracy": 0.8})
            current = json.loads(path.read_text())

            self.assertEqual(len(current["audit_hashes"]), 2)
            self.assertNotEqual(current["audit_hashes"][0], current["audit_hashes"][1])

    def test_catboost_native_serialization_round_trip(self):
        x = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.2, 0.1],
                [1.0, 1.0],
                [1.1, 1.0],
                [1.0, 1.1],
            ],
            dtype=float,
        )
        y = np.array([0, 0, 0, 1, 1, 1])
        model = CatBoostModel(n_features=2, iterations=4, depth=2)
        model.train_epochs(x, y, cfg=None)
        params = model.get_parameters()

        restored = CatBoostModel(n_features=2, iterations=4, depth=2)
        restored.set_parameters(params)

        self.assertEqual(params[0].dtype, np.uint8)
        self.assertGreater(len(params[0]), 0)
        self.assertTrue(np.allclose(model.predict_proba(x), restored.predict_proba(x)))

    def test_resource_presets_apply_model_specific_defaults(self):
        dp_config = resolve_resource_config(preset="dev", model_name="dp-mlp")
        quantum_config = resolve_resource_config(preset="dev", model_name="hybrid-quantum")
        override_config = resolve_resource_config(
            preset="quick",
            model_name="logreg",
            num_clients=6,
            num_rounds=7,
        )

        self.assertEqual(dp_config["num_clients"], 2)
        self.assertEqual(dp_config["num_rounds"], 3)
        self.assertEqual(dp_config["batch_size"], 64)
        self.assertEqual(quantum_config["num_clients"], 2)
        self.assertEqual(quantum_config["batch_size"], 32)
        self.assertEqual(override_config["num_clients"], 6)
        self.assertEqual(override_config["num_rounds"], 7)

    def test_label_flip_helper_only_changes_requested_clients(self):
        partitions = {
            0: DatasetPartition(
                train_x=np.zeros((2, 1)),
                train_y=np.array([0, 1]),
                val_x=np.zeros((2, 1)),
                val_y=np.array([1, 0]),
            ),
            1: DatasetPartition(
                train_x=np.zeros((2, 1)),
                train_y=np.array([1, 1]),
                val_x=np.zeros((2, 1)),
                val_y=np.array([0, 0]),
            ),
        }

        flipped = clone_partitions_with_label_flip(partitions, malicious_clients=1)

        self.assertTrue(np.array_equal(flipped[0].train_y, np.array([1, 0])))
        self.assertTrue(np.array_equal(flipped[0].val_y, np.array([0, 1])))
        self.assertTrue(np.array_equal(flipped[1].train_y, partitions[1].train_y))


if __name__ == "__main__":
    unittest.main()
