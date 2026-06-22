import unittest
import numpy as np
from federated_malware.strategy.aggregators import (
    clip_update, apply_topk, apply_quantization, apply_server_dp,
    is_outlier, record_update,
)


class TransformTest(unittest.TestCase):

    def test_clip_update_scales_down_large_norms(self):
        update = [np.array([3.0, 4.0])]  # norm = 5.0
        clipped = clip_update(update, max_update_norm=1.0)
        self.assertLessEqual(np.linalg.norm(clipped[0]), 1.0 + 1e-9)

    def test_clip_update_noop_when_within_norm(self):
        update = [np.array([0.3, 0.4])]  # norm = 0.5
        clipped = clip_update(update, max_update_norm=1.0)
        np.testing.assert_array_equal(clipped[0], update[0])

    def test_clip_update_noop_when_none(self):
        update = [np.array([3.0, 4.0])]
        clipped = clip_update(update, max_update_norm=None)
        np.testing.assert_array_equal(clipped[0], update[0])

    def test_apply_topk_keeps_top_entries(self):
        update = [np.array([10.0, 1.0, 0.5, 0.1])]
        sparse = apply_topk(update, topk_ratio=0.5)  # keep top 2
        self.assertEqual(int(np.count_nonzero(sparse[0])), 2)
        # The largest values should be preserved
        self.assertAlmostEqual(sparse[0][0], 10.0)

    def test_apply_topk_noop_when_none(self):
        update = [np.array([1.0, 2.0])]
        result = apply_topk(update, topk_ratio=None)
        np.testing.assert_array_equal(result[0], update[0])

    def test_apply_topk_invalid_ratio_raises(self):
        with self.assertRaises(ValueError):
            apply_topk([np.array([1.0])], topk_ratio=0.0)
        with self.assertRaises(ValueError):
            apply_topk([np.array([1.0])], topk_ratio=1.5)

    def test_apply_quantization_preserves_shape(self):
        update = [np.array([1.0, 2.5, 3.7, 0.1])]
        quantized = apply_quantization(update, quantization_bits=8)
        self.assertEqual(quantized[0].shape, update[0].shape)

    def test_apply_quantization_noop_when_none(self):
        update = [np.array([1.0, 2.0])]
        result = apply_quantization(update, quantization_bits=None)
        np.testing.assert_array_equal(result[0], update[0])

    def test_apply_quantization_invalid_bits_raises(self):
        with self.assertRaises(ValueError):
            apply_quantization([np.array([1.0])], quantization_bits=16)

    def test_apply_quantization_constant_array_is_identity(self):
        update = [np.array([5.0, 5.0, 5.0])]
        result = apply_quantization(update, quantization_bits=8)
        np.testing.assert_array_equal(result[0], update[0])

    def test_apply_server_dp_adds_noise(self):
        rng = np.random.default_rng(42)
        update = [np.array([1.0, 2.0, 3.0])]
        noisy = apply_server_dp(update, num_clients=5, server_dp_noise=1.0, max_update_norm=1.0, rng=rng)
        # Should not be exactly equal (noise added)
        self.assertFalse(np.array_equal(noisy[0], update[0]))

    def test_apply_server_dp_noop_when_zero(self):
        rng = np.random.default_rng(42)
        update = [np.array([1.0, 2.0])]
        result = apply_server_dp(update, num_clients=5, server_dp_noise=0.0, max_update_norm=1.0, rng=rng)
        np.testing.assert_array_equal(result[0], update[0])

    def test_is_outlier_returns_false_with_short_history(self):
        history = {"c0": [1.0, 1.0]}  # less than 3 entries
        result = is_outlier("c0", [np.array([1.0])], flanders_z=2.0, history=history)
        self.assertFalse(result)

    def test_is_outlier_detects_large_deviation(self):
        # Need std > 0 so the z-score check can fire; use varied history.
        history = {"c0": [1.0, 2.0, 1.5, 1.0, 2.0]}  # mean≈1.5, std≈0.45
        # A huge update → norm=100, z = |100 - 1.5| / 0.45 ≫ 2
        result = is_outlier("c0", [np.array([100.0])], flanders_z=2.0, history=history)
        self.assertTrue(result)

    def test_is_outlier_noop_when_z_is_none(self):
        history = {"c0": [1.0, 2.0, 1.5, 1.0]}
        result = is_outlier("c0", [np.array([100.0])], flanders_z=None, history=history)
        self.assertFalse(result)

    def test_record_update_appends_norm(self):
        history = {}
        record_update("c0", [np.array([3.0, 4.0])], history)
        self.assertIn("c0", history)
        self.assertEqual(len(history["c0"]), 1)
        self.assertAlmostEqual(history["c0"][0], 5.0)

    def test_record_update_accumulates(self):
        history = {"c0": [1.0]}
        record_update("c0", [np.array([3.0, 4.0])], history)
        self.assertEqual(len(history["c0"]), 2)


if __name__ == "__main__":
    unittest.main()
