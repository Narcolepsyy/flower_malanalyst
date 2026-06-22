import unittest
import numpy as np
from federated_malware.strategy.aggregators import (
    agg_fedavg, agg_median, agg_trimmed_mean, agg_krum, agg_bulyan,
    agg_median_of_means, krum_index,
)


class AggregatorTest(unittest.TestCase):

    def test_agg_fedavg_weighted_mean(self):
        """FedAvg should produce weighted average."""
        params = [
            [np.array([1.0, 0.0])],
            [np.array([0.0, 1.0])],
        ]
        weights = np.array([3.0, 1.0])
        result = agg_fedavg(params, weights)
        np.testing.assert_allclose(result[0], np.array([0.75, 0.25]))

    def test_agg_fedavg_equal_weights(self):
        params = [[np.array([2.0])], [np.array([4.0])]]
        weights = np.array([1.0, 1.0])
        result = agg_fedavg(params, weights)
        np.testing.assert_allclose(result[0], np.array([3.0]))

    def test_agg_median_picks_coordinate_wise_median(self):
        params = [
            [np.array([1.0, 10.0])],
            [np.array([2.0, 20.0])],
            [np.array([100.0, 30.0])],  # outlier in dim 0
        ]
        result = agg_median(params)
        np.testing.assert_allclose(result[0], np.array([2.0, 20.0]))

    def test_agg_trimmed_mean_excludes_extremes(self):
        params = [
            [np.array([1.0])],
            [np.array([2.0])],
            [np.array([3.0])],
            [np.array([100.0])],  # outlier
        ]
        result = agg_trimmed_mean(params, trim_ratio=0.25)  # trim 1 from each end
        # After trimming: [2.0, 3.0] → mean = 2.5
        np.testing.assert_allclose(result[0], np.array([2.5]))

    def test_agg_trimmed_mean_falls_back_to_median_when_overtrimmed(self):
        params = [[np.array([1.0])], [np.array([2.0])]]
        # trim_ratio=0.5 means k=1, 2*k >= n → fallback to median
        result = agg_trimmed_mean(params, trim_ratio=0.5)
        np.testing.assert_allclose(result[0], np.array([1.5]))

    def test_krum_index_picks_closest_to_neighbours(self):
        params = [
            [np.array([1.0, 1.0])],
            [np.array([1.1, 0.9])],
            [np.array([100.0, 100.0])],  # outlier
        ]
        winner = krum_index(params, f=1)
        self.assertIn(winner, [0, 1])  # should not pick outlier

    def test_agg_krum_returns_winner_params(self):
        params = [
            [np.array([1.0, 1.0])],
            [np.array([1.1, 0.9])],
            [np.array([100.0, 100.0])],
        ]
        result = agg_krum(params, f=1)
        # Should return one of the non-outlier params
        norm = np.linalg.norm(result[0] - np.array([1.0, 1.0]))
        self.assertLess(norm, 1.0)

    def test_agg_krum_single_client(self):
        params = [[np.array([5.0])]]
        result = agg_krum(params, f=0)
        np.testing.assert_allclose(result[0], np.array([5.0]))

    def test_agg_bulyan_resists_outlier(self):
        params = [
            [np.array([1.0])],
            [np.array([1.1])],
            [np.array([0.9])],
            [np.array([1.0])],
            [np.array([100.0])],  # Byzantine
        ]
        result = agg_bulyan(params, f=1)
        self.assertLess(abs(result[0][0] - 1.0), 0.2)

    def test_agg_median_of_means_averages_group_medians(self):
        params = [
            [np.array([1.0])],
            [np.array([1.0])],
            [np.array([3.0])],
            [np.array([3.0])],
        ]
        result = agg_median_of_means(params, mom_groups=2)
        # Group 1: mean([1,1])=1, Group 2: mean([3,3])=3 → median = 2.0
        np.testing.assert_allclose(result[0], np.array([2.0]))

    def test_agg_median_multi_layer(self):
        """Verify aggregation works with multi-layer parameter sets."""
        params = [
            [np.array([1.0]), np.array([10.0, 20.0])],
            [np.array([2.0]), np.array([30.0, 40.0])],
            [np.array([3.0]), np.array([50.0, 60.0])],
        ]
        result = agg_median(params)
        np.testing.assert_allclose(result[0], np.array([2.0]))
        np.testing.assert_allclose(result[1], np.array([30.0, 40.0]))


if __name__ == "__main__":
    unittest.main()
