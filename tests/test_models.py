import unittest
import numpy as np
from federated_malware.models.base import TrainConfig
from federated_malware.models.logreg import NumpyLogisticModel
from federated_malware.models.mlp import TorchMLPModel


def _make_separable_data(n=200, n_features=10, seed=42):
    """Create a linearly separable binary classification dataset."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_features))
    w = rng.standard_normal(n_features)
    y = (x @ w > 0).astype(np.int64)
    return x.astype(np.float64), y


class LogregModelTest(unittest.TestCase):

    def test_parameter_round_trip(self):
        """get_parameters → set_parameters → predict_proba should be identical."""
        model = NumpyLogisticModel(n_features=5, seed=42)
        x = np.random.default_rng(0).standard_normal((10, 5))
        params = model.get_parameters()
        proba_before = model.predict_proba(x)

        # Create a new model and load the same params
        model2 = NumpyLogisticModel(n_features=5, seed=99)  # different seed
        model2.set_parameters(params)
        proba_after = model2.predict_proba(x)
        np.testing.assert_allclose(proba_before, proba_after)

    def test_train_reduces_loss(self):
        """Training on a separable dataset should reduce loss."""
        x, y = _make_separable_data(n=200, n_features=10)
        model = NumpyLogisticModel(n_features=10, lr=0.1, seed=42)
        cfg = TrainConfig(lr=0.1, epochs=1, batch_size=64, seed=42)

        metrics_before = model.evaluate(x, y)
        model.train_epochs(x, y, cfg)
        metrics_after = model.evaluate(x, y)
        self.assertLess(metrics_after["loss"], metrics_before["loss"])

    def test_evaluate_on_empty_data(self):
        model = NumpyLogisticModel(n_features=5, seed=42)
        metrics = model.evaluate(np.empty((0, 5)), np.empty(0, dtype=np.int64))
        self.assertEqual(metrics["loss"], 0.0)


class MLPModelTest(unittest.TestCase):

    def test_parameter_round_trip(self):
        model = TorchMLPModel(n_features=5, hidden1=16, hidden2=8, seed=42)
        x = np.random.default_rng(0).standard_normal((10, 5)).astype(np.float32)
        params = model.get_parameters()
        proba_before = model.predict_proba(x)

        model2 = TorchMLPModel(n_features=5, hidden1=16, hidden2=8, seed=99)
        model2.set_parameters(params)
        proba_after = model2.predict_proba(x)
        np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)

    def test_train_reduces_loss(self):
        x, y = _make_separable_data(n=200, n_features=10)
        x = x.astype(np.float32)
        model = TorchMLPModel(n_features=10, hidden1=32, hidden2=16, lr=1e-2, seed=42)
        cfg = TrainConfig(lr=1e-2, epochs=5, batch_size=64, seed=42)

        metrics_before = model.evaluate(x, y)
        model.train_epochs(x, y, cfg)
        metrics_after = model.evaluate(x, y)
        self.assertLess(metrics_after["loss"], metrics_before["loss"])


if __name__ == "__main__":
    unittest.main()
