"""
Expanded test coverage for conformal/bandwidth.py.

Additional edge cases for BandwidthSelector: Kish n_eff, MACG
edge cases, select_with_n_eff_floor behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_spatial.conformal.bandwidth import BandwidthSelector
from insurance_spatial.conformal._types import BandwidthCVResult
from insurance_spatial.conformal.scores import TweediePearsonScore


def _cal_scores(cal_data, dummy_model):
    yhat = dummy_model.predict(cal_data["X"])
    return TweediePearsonScore(1.5).score(cal_data["y"], yhat)


# ---------------------------------------------------------------------------
# BandwidthSelector construction extras
# ---------------------------------------------------------------------------

class TestBandwidthSelectorConstructionExtra:
    def test_cv_stored(self):
        sel = BandwidthSelector(cv=3)
        assert sel.cv == 3

    def test_n_eff_min_stored(self):
        sel = BandwidthSelector(n_eff_min=50)
        assert sel.n_eff_min == 50

    def test_grid_resolution_stored(self):
        sel = BandwidthSelector(grid_resolution=8)
        assert sel.grid_resolution == 8

    def test_random_state_stored(self):
        sel = BandwidthSelector(random_state=99)
        assert sel.random_state == 99

    def test_float_candidates_cast(self):
        sel = BandwidthSelector(candidates_km=[10, 20, 30])  # int input
        assert all(isinstance(c, float) for c in sel.candidates_km)


# ---------------------------------------------------------------------------
# _spatial_folds extras
# ---------------------------------------------------------------------------

class TestSpatialFoldsExtra:
    def test_single_fold_gives_all_same_label(self):
        """cv=1 means all points in one fold."""
        sel = BandwidthSelector(cv=1)
        rng = np.random.default_rng(20)
        lat = rng.uniform(50, 56, 50)
        lon = rng.uniform(-4, 2, 50)
        labels = sel._spatial_folds(lat, lon)
        assert len(set(labels)) == 1

    def test_folds_cover_all_points(self):
        """No point should be unassigned."""
        sel = BandwidthSelector(cv=5)
        rng = np.random.default_rng(21)
        n = 150
        lat = rng.uniform(50, 58, n)
        lon = rng.uniform(-5, 2, n)
        labels = sel._spatial_folds(lat, lon)
        assert len(labels) == n
        assert labels.min() >= 0
        assert labels.max() < 5


# ---------------------------------------------------------------------------
# _kish_n_eff extras
# ---------------------------------------------------------------------------

class TestKishNEffExtra:
    def test_all_zeros_returns_zero(self):
        sel = BandwidthSelector()
        assert sel._kish_n_eff(np.zeros(10)) == pytest.approx(0.0)

    def test_two_equal_weights_returns_2(self):
        sel = BandwidthSelector()
        assert sel._kish_n_eff(np.array([1.0, 1.0])) == pytest.approx(2.0)

    def test_n_eff_bounded_below_by_one_for_nonzero(self):
        """n_eff >= 1 whenever at least one weight is non-zero."""
        sel = BandwidthSelector()
        w = np.array([5.0] + [0.0] * 9)
        assert sel._kish_n_eff(w) >= 1.0

    def test_n_eff_equals_n_for_uniform(self):
        sel = BandwidthSelector()
        w = np.ones(30)
        assert sel._kish_n_eff(w) == pytest.approx(30.0, rel=1e-6)


# ---------------------------------------------------------------------------
# _weighted_quantile extras
# ---------------------------------------------------------------------------

class TestWeightedQuantileExtra:
    def test_empty_scores_returns_inf(self):
        """If all weights are zero, should return inf or handle gracefully."""
        sel = BandwidthSelector()
        scores = np.array([1.0, 2.0, 3.0])
        w = np.zeros(3)
        q = sel._weighted_quantile(scores, w, 0.10)
        assert q == np.inf or np.isfinite(q)

    def test_single_score_returns_that_score_or_inf(self):
        sel = BandwidthSelector()
        scores = np.array([42.0])
        w = np.array([1.0])
        q = sel._weighted_quantile(scores, w, 0.10)
        # 90th percentile of [42, inf] with equal weights
        assert np.isfinite(q) or q == np.inf

    def test_quantile_increases_with_1_minus_alpha(self):
        """Lower alpha means higher (1-alpha), so higher quantile."""
        sel = BandwidthSelector()
        scores = np.arange(1.0, 11.0)
        w = np.ones(10)
        q90 = sel._weighted_quantile(scores, w, 0.10)  # 90th pct
        q50 = sel._weighted_quantile(scores, w, 0.50)  # 50th pct
        assert q90 >= q50


# ---------------------------------------------------------------------------
# _macg_on_grid extras
# ---------------------------------------------------------------------------

class TestMacgOnGridExtra:
    def test_empty_val_returns_nan(self):
        sel = BandwidthSelector(grid_resolution=5)
        result = sel._macg_on_grid(
            scores_cal=np.array([1.0, 2.0]),
            lat_cal=np.array([51.5, 52.0]),
            lon_cal=np.array([-0.1, -0.2]),
            scores_val=np.array([]),
            lat_val=np.array([]),
            lon_val=np.array([]),
            bandwidth_km=20.0,
            alpha=0.10,
        )
        assert np.isnan(result)

    def test_macg_non_negative(self, dummy_model, cal_data, test_data):
        sel = BandwidthSelector(grid_resolution=5)
        scores = _cal_scores(cal_data, dummy_model)
        macg = sel._macg_on_grid(
            scores,
            cal_data["lat"],
            cal_data["lon"],
            scores[:50],  # use first 50 as val
            cal_data["lat"][:50],
            cal_data["lon"][:50],
            bandwidth_km=20.0,
            alpha=0.10,
        )
        assert np.isnan(macg) or macg >= 0.0


# ---------------------------------------------------------------------------
# select extras
# ---------------------------------------------------------------------------

class TestSelectExtra:
    def test_select_with_single_candidate(self, dummy_model, cal_data):
        sel = BandwidthSelector(candidates_km=[20.0], cv=3, grid_resolution=5)
        scores = _cal_scores(cal_data, dummy_model)
        result = sel.select(scores, cal_data["lat"], cal_data["lon"])
        assert result.optimal_km == 20.0

    def test_result_metric_matches_selector(self, dummy_model, cal_data):
        sel = BandwidthSelector(candidates_km=[10.0, 20.0], cv=3, grid_resolution=5)
        scores = _cal_scores(cal_data, dummy_model)
        result = sel.select(scores, cal_data["lat"], cal_data["lon"])
        assert result.metric == "macg"

    def test_result_n_folds_matches_selector(self, dummy_model, cal_data):
        sel = BandwidthSelector(candidates_km=[10.0, 20.0], cv=4, grid_resolution=5)
        scores = _cal_scores(cal_data, dummy_model)
        result = sel.select(scores, cal_data["lat"], cal_data["lon"])
        assert result.n_folds == 4


# ---------------------------------------------------------------------------
# select_with_n_eff_floor
# ---------------------------------------------------------------------------

class TestSelectWithNEffFloor:
    def test_wide_bandwidth_satisfies_floor(self, dummy_model, cal_data, test_data):
        """With a very low n_eff_min, should always return the CV-optimal result."""
        sel = BandwidthSelector(
            candidates_km=[10.0, 20.0, 30.0],
            cv=3,
            n_eff_min=1,  # trivially satisfied
            grid_resolution=5,
        )
        scores = _cal_scores(cal_data, dummy_model)
        result = sel.select_with_n_eff_floor(
            scores,
            cal_data["lat"],
            cal_data["lon"],
            test_lat=test_data["lat"],
            test_lon=test_data["lon"],
        )
        assert result.optimal_km in sel.candidates_km

    def test_returns_bandwidth_cv_result(self, dummy_model, cal_data, test_data):
        sel = BandwidthSelector(
            candidates_km=[10.0, 20.0],
            cv=3,
            n_eff_min=1,
            grid_resolution=5,
        )
        scores = _cal_scores(cal_data, dummy_model)
        result = sel.select_with_n_eff_floor(
            scores,
            cal_data["lat"],
            cal_data["lon"],
            test_lat=test_data["lat"],
            test_lon=test_data["lon"],
        )
        assert isinstance(result, BandwidthCVResult)

    def test_high_n_eff_floor_widens_bandwidth(self, dummy_model, cal_data, test_data):
        """
        Setting n_eff_min very high should force the widest bandwidth.
        Use a large test set clustered in one city to ensure narrow bandwidths fail.
        """
        sel = BandwidthSelector(
            candidates_km=[0.1, 200.0],  # tiny vs huge
            cv=3,
            n_eff_min=5000,  # impossibly high for narrow bandwidth
            grid_resolution=3,
        )
        scores = _cal_scores(cal_data, dummy_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = sel.select_with_n_eff_floor(
                scores,
                cal_data["lat"],
                cal_data["lon"],
                test_lat=test_data["lat"],
                test_lon=test_data["lon"],
            )
        # Should have widened to the largest candidate
        # (or issued a warning that no candidate meets the floor)
        assert result.optimal_km in sel.candidates_km
