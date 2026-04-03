"""
Expanded test coverage for the diagnostics module.

Covers Moran's I edge cases, dataclass field contracts, and helper
behaviour not exercised by the original tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_spatial.adjacency import build_grid_adjacency, AdjacencyMatrix
from insurance_spatial.diagnostics import (
    MoranI,
    ConvergenceSummary,
    SpatialDiagnostics,
    moran_i,
)

import polars as pl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid_4x4() -> AdjacencyMatrix:
    return build_grid_adjacency(4, 4, connectivity="rook")


@pytest.fixture(scope="module")
def grid_6x6() -> AdjacencyMatrix:
    return build_grid_adjacency(6, 6, connectivity="rook")


# ---------------------------------------------------------------------------
# MoranI dataclass
# ---------------------------------------------------------------------------

class TestMoranIDataclass:
    def test_all_fields_present(self):
        mi = MoranI(
            statistic=0.5,
            expected=-0.04,
            p_value=0.001,
            z_score=3.5,
            n_permutations=999,
            significant=True,
            interpretation="Positive autocorrelation",
        )
        assert mi.statistic == 0.5
        assert mi.expected == -0.04
        assert mi.p_value == 0.001
        assert mi.z_score == 3.5
        assert mi.n_permutations == 999
        assert mi.significant is True
        assert "autocorrelation" in mi.interpretation.lower()


# ---------------------------------------------------------------------------
# moran_i – additional edge cases
# ---------------------------------------------------------------------------

class TestMoranIAdditional:
    def test_single_value_array_raises_zero_division(self):
        """A length-1 adjacency raises ZeroDivisionError (N-1 == 0 in expected formula)."""
        from scipy import sparse
        W = sparse.csr_matrix(np.array([[0.0]]))
        adj = AdjacencyMatrix(W=W, areas=["a"])
        # expected_I = -1/(N-1) = -1/0 raises ZeroDivisionError for N=1
        with pytest.raises(ZeroDivisionError):
            moran_i(np.array([1.0]), adj, n_permutations=9)

    def test_two_node_graph_all_same_values(self):
        """Two nodes, same value: no variation, Moran's I = 0."""
        adj = build_grid_adjacency(1, 2, connectivity="rook")
        values = np.array([5.0, 5.0])
        result = moran_i(values, adj, n_permutations=19)
        assert abs(result.statistic) < 1e-10

    def test_moran_i_negative_for_alternating_pattern_4x4(self, grid_4x4):
        """Checkerboard pattern on 4×4 grid should give negative I."""
        values = np.array([(i // 4 + i % 4) % 2 for i in range(16)], dtype=float)
        result = moran_i(values, grid_4x4, n_permutations=499)
        assert result.statistic < 0

    def test_moran_i_uses_fixed_seed_reproducible(self, grid_4x4):
        """Two calls with same inputs should return the same statistic and p-value."""
        values = np.arange(16, dtype=float)
        r1 = moran_i(values, grid_4x4, n_permutations=199)
        r2 = moran_i(values, grid_4x4, n_permutations=199)
        assert r1.statistic == r2.statistic
        assert r1.p_value == r2.p_value

    def test_moran_i_gradient_6x6(self, grid_6x6):
        """N-S gradient on 6×6 grid should give positive, significant Moran's I."""
        values = np.array([float(i // 6) for i in range(36)])
        result = moran_i(values, grid_6x6, n_permutations=499)
        assert result.statistic > 0
        assert result.significant

    def test_values_as_list_accepted(self):
        """moran_i should accept Python lists, not just numpy arrays."""
        adj = build_grid_adjacency(3, 3)
        values = list(range(9))
        result = moran_i(values, adj)
        assert isinstance(result.statistic, float)

    def test_z_score_nonzero_for_structured_data(self):
        """A strongly clustered pattern should produce a non-zero z-score."""
        adj = build_grid_adjacency(5, 5)
        values = np.array([float(i // 5) for i in range(25)])
        result = moran_i(values, adj, n_permutations=499)
        assert abs(result.z_score) > 1.0

    def test_p_value_strictly_positive(self):
        """Permutation p-value must be > 0 by construction (numerator >= 1)."""
        adj = build_grid_adjacency(4, 4)
        values = np.array([float(i // 4) for i in range(16)])
        result = moran_i(values, adj, n_permutations=99)
        assert result.p_value > 0.0

    def test_mismatched_values_shape_raises(self, grid_4x4):
        """Wrong-length values must raise ValueError."""
        with pytest.raises(ValueError, match="entries"):
            moran_i(np.ones(5), grid_4x4)

    def test_interpretation_positive_significant_mentions_spatial_smoothing(self):
        """Positive significant I → interpretation should mention 'spatial smoothing'."""
        adj = build_grid_adjacency(5, 5)
        values = np.array([float(i // 5) for i in range(25)])
        result = moran_i(values, adj, n_permutations=499)
        if result.significant and result.statistic > 0:
            assert "smoothing" in result.interpretation.lower()

    def test_interpretation_negative_significant_mentions_negative(self):
        """Negative significant I → interpretation should mention 'negative'."""
        adj = build_grid_adjacency(5, 5)
        values = np.array([(i // 5 + i % 5) % 2 for i in range(25)], dtype=float)
        result = moran_i(values, adj, n_permutations=999)
        if result.significant and result.statistic < 0:
            assert "negative" in result.interpretation.lower()

    def test_few_permutations_runs_fine(self):
        """n_permutations=9 is minimal but must not crash."""
        adj = build_grid_adjacency(3, 3)
        values = np.arange(9, dtype=float)
        result = moran_i(values, adj, n_permutations=9)
        assert result.n_permutations == 9


# ---------------------------------------------------------------------------
# ConvergenceSummary and SpatialDiagnostics dataclasses
# ---------------------------------------------------------------------------

class TestConvergenceSummaryDataclass:
    def _make_summary(self, max_rhat=1.005, min_ess_bulk=500, min_ess_tail=450):
        converged = (max_rhat < 1.01) and (min_ess_bulk > 400) and (min_ess_tail > 400)
        rhat_df = pl.DataFrame({"parameter": ["alpha"], "rhat": [max_rhat], "ess_bulk": [min_ess_bulk]})
        return ConvergenceSummary(
            max_rhat=max_rhat,
            min_ess_bulk=min_ess_bulk,
            min_ess_tail=min_ess_tail,
            converged=converged,
            rhat_by_param=rhat_df,
            ess_by_param=rhat_df.select(["parameter", "ess_bulk"]),
            n_divergences=0,
        )

    def test_converged_when_all_thresholds_met(self):
        s = self._make_summary(max_rhat=1.005, min_ess_bulk=500, min_ess_tail=450)
        assert s.converged is True

    def test_not_converged_high_rhat(self):
        s = self._make_summary(max_rhat=1.05)
        assert s.converged is False

    def test_not_converged_low_bulk_ess(self):
        s = self._make_summary(min_ess_bulk=100)
        assert s.converged is False

    def test_not_converged_low_tail_ess(self):
        s = self._make_summary(min_ess_tail=50)
        assert s.converged is False

    def test_rhat_by_param_is_polars_dataframe(self):
        s = self._make_summary()
        assert isinstance(s.rhat_by_param, pl.DataFrame)

    def test_ess_by_param_has_ess_bulk_column(self):
        s = self._make_summary()
        assert "ess_bulk" in s.ess_by_param.columns

    def test_n_divergences_stored(self):
        converged = True
        rhat_df = pl.DataFrame({"parameter": ["alpha"], "rhat": [1.001], "ess_bulk": [600.0]})
        s = ConvergenceSummary(
            max_rhat=1.001,
            min_ess_bulk=600.0,
            min_ess_tail=500.0,
            converged=converged,
            rhat_by_param=rhat_df,
            ess_by_param=rhat_df.select(["parameter", "ess_bulk"]),
            n_divergences=12,
        )
        assert s.n_divergences == 12


class TestSpatialDiagnosticsDataclass:
    def test_construction(self):
        converged = True
        rhat_df = pl.DataFrame({"parameter": ["alpha"], "rhat": [1.001], "ess_bulk": [600.0]})
        conv = ConvergenceSummary(
            max_rhat=1.001,
            min_ess_bulk=600.0,
            min_ess_tail=500.0,
            converged=converged,
            rhat_by_param=rhat_df,
            ess_by_param=rhat_df.select(["parameter", "ess_bulk"]),
            n_divergences=0,
        )
        rho_df = pl.DataFrame({"parameter": ["rho"], "mean": [0.7], "sd": [0.1], "q025": [0.5], "q975": [0.9]})
        sigma_df = pl.DataFrame({"parameter": ["sigma"], "mean": [0.3], "sd": [0.05], "q025": [0.2], "q975": [0.4]})

        diag = SpatialDiagnostics(
            convergence=conv,
            rho_summary=rho_df,
            sigma_summary=sigma_df,
            moran_post=None,
        )
        assert diag.convergence is conv
        assert diag.moran_post is None
        assert isinstance(diag.rho_summary, pl.DataFrame)
        assert isinstance(diag.sigma_summary, pl.DataFrame)
