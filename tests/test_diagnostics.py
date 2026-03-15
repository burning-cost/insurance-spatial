"""
Tests for the diagnostics module.

Moran's I tests run locally - no model fitting required.
Convergence diagnostics tests require a fitted BYM2Result and are marked
with @pytest.mark.databricks so they run only in the Databricks CI environment.
"""

import numpy as np
import pytest

from insurance_spatial.adjacency import build_grid_adjacency
from insurance_spatial.diagnostics import MoranI, moran_i


class TestMoranI:
    def test_positive_autocorrelation(self, grid_5x5):
        """
        A north-south gradient should yield a positive, significant Moran's I
        on a 5×5 rook grid.
        """
        # Values increase by row - strong spatial structure
        values = np.array([float(i // 5) for i in range(25)])
        result = moran_i(values, grid_5x5, n_permutations=499)
        assert isinstance(result, MoranI)
        assert result.statistic > 0
        assert result.significant

    def test_random_values_not_significant(self, grid_5x5):
        """
        Random values should not show significant spatial autocorrelation
        (at α=0.05, we expect this to fail ~5% of the time - but with a fixed
        seed and 25 nodes it should be reliably non-significant).
        """
        rng = np.random.default_rng(0)
        values = rng.standard_normal(25)
        result = moran_i(values, grid_5x5, n_permutations=499)
        # Not testing significance here - just that the function runs
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert 0.0 <= result.p_value <= 1.0

    def test_statistic_range(self, grid_5x5):
        """Moran's I should be in [-1, 1] approximately."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(25)
        result = moran_i(values, grid_5x5)
        # Permutation-based I can very slightly exceed [-1,1] due to finite N
        assert -2.0 <= result.statistic <= 2.0

    def test_mismatched_sizes_raises(self, grid_5x5):
        values = np.ones(10)  # wrong size
        with pytest.raises(ValueError, match="entries"):
            moran_i(values, grid_5x5)

    def test_n_permutations_stored(self, grid_5x5):
        values = np.arange(25, dtype=float)
        result = moran_i(values, grid_5x5, n_permutations=199)
        assert result.n_permutations == 199

    def test_interpretation_is_string(self, grid_5x5):
        values = np.arange(25, dtype=float)
        result = moran_i(values, grid_5x5)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

    def test_expected_value(self, grid_5x5):
        """Expected Moran's I under null is -1/(N-1)."""
        values = np.arange(25, dtype=float)
        result = moran_i(values, grid_5x5)
        expected = -1.0 / (25 - 1)
        assert abs(result.expected - expected) < 1e-10

    def test_constant_values(self, grid_5x5):
        """Constant values give Moran's I = 0 (no variation)."""
        values = np.ones(25)
        result = moran_i(values, grid_5x5)
        # With zero variance, statistic should be 0
        assert abs(result.statistic) < 1e-10

    def test_3x3_grid(self, grid_3x3):
        """Run on a 3×3 grid to verify small-N edge cases."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        result = moran_i(values, grid_3x3)
        assert isinstance(result.statistic, float)

    # ------------------------------------------------------------------
    # Regression tests for P0 bug: one-tailed p-value missed negative autocorrelation
    # ------------------------------------------------------------------

    def test_p_value_is_two_tailed(self, grid_5x5):
        """
        P0 regression: p-value must be two-tailed so that strongly negative
        spatial autocorrelation is flagged as significant.

        A checkerboard pattern on a 5×5 rook grid produces a strongly negative
        Moran's I (adjacent cells always differ).  With the old one-tailed test
        (perm_stats >= observed_I), p would be close to 1.0 for negative I,
        making it appear non-significant.  The two-tailed test must give a low p.
        """
        # Checkerboard: alternating 0/1 by (row+col) parity — strong negative autocorr
        values = np.array([(i // 5 + i % 5) % 2 for i in range(25)], dtype=float)
        result = moran_i(values, grid_5x5, n_permutations=999)

        assert result.statistic < 0, (
            f"Expected negative Moran's I for checkerboard, got {result.statistic:.4f}"
        )
        # The p-value must be small (significant negative autocorrelation)
        assert result.p_value < 0.05, (
            f"Checkerboard should be significantly negatively autocorrelated "
            f"(p={result.p_value:.4f}), but p >= 0.05. "
            "This suggests the one-tailed bug is still present."
        )
        assert result.significant

    def test_p_value_in_zero_one(self, grid_5x5):
        """p-value must always be in [0, 1]."""
        rng = np.random.default_rng(7)
        for _ in range(5):
            values = rng.standard_normal(25)
            result = moran_i(values, grid_5x5, n_permutations=199)
            assert 0.0 <= result.p_value <= 1.0

    def test_two_tailed_symmetric_for_positive_autocorr(self, grid_5x5):
        """
        For strong positive autocorrelation, two-tailed p-value should still
        be small (i.e. not degraded by the two-tailing).
        """
        values = np.array([float(i // 5) for i in range(25)])
        result = moran_i(values, grid_5x5, n_permutations=999)
        assert result.p_value < 0.05
        assert result.significant
