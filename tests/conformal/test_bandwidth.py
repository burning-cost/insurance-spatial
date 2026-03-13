"""
Tests for bandwidth.py: BandwidthSelector.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_spatial.conformal.bandwidth import BandwidthSelector
from insurance_spatial.conformal._types import BandwidthCVResult
from insurance_spatial.conformal.scores import TweediePearsonScore


def _make_scores(dummy_model, cal_data):
    """Helper: compute calibration scores using the dummy model."""
    yhat = dummy_model.predict(cal_data["X"])
    return TweediePearsonScore(1.5).score(cal_data["y"], yhat)


class TestBandwidthSelectorConstruction:
    def test_defaults(self):
        sel = BandwidthSelector()
        assert sel.cv == 5
        assert sel.n_eff_min == 30
        assert len(sel.candidates_km) > 0
        assert sel.metric == "macg"

    def test_custom_candidates(self):
        sel = BandwidthSelector(candidates_km=[10.0, 20.0, 30.0])
        assert sel.candidates_km == [10.0, 20.0, 30.0]

    def test_candidates_sorted(self):
        sel = BandwidthSelector(candidates_km=[30.0, 5.0, 15.0])
        assert sel.candidates_km == sorted([30.0, 5.0, 15.0])

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="macg"):
            BandwidthSelector(metric="rmse")


class TestSpatialFolds:
    def test_fold_labels_in_range(self):
        rng = np.random.default_rng(0)
        lat = rng.uniform(50, 56, 100)
        lon = rng.uniform(-4, 2, 100)
        sel = BandwidthSelector(cv=5)
        labels = sel._spatial_folds(lat, lon)
        assert labels.shape == (100,)
        assert set(labels).issubset(set(range(5)))

    def test_all_folds_represented(self):
        rng = np.random.default_rng(1)
        lat = rng.uniform(50, 56, 200)
        lon = rng.uniform(-4, 2, 200)
        sel = BandwidthSelector(cv=5)
        labels = sel._spatial_folds(lat, lon)
        assert len(set(labels)) == 5


class TestKishNEff:
    def test_uniform_weights(self):
        sel = BandwidthSelector()
        w = np.ones(50)
        assert sel._kish_n_eff(w) == pytest.approx(50.0, rel=1e-4)

    def test_single_dominant_weight(self):
        sel = BandwidthSelector()
        w = np.array([1000.0] + [0.001] * 99)
        assert sel._kish_n_eff(w) < 5.0


class TestWeightedQuantile:
    def test_sorted_scores_monotone(self):
        sel = BandwidthSelector()
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones(5)
        q10 = sel._weighted_quantile(scores, w, 0.10)
        q50 = sel._weighted_quantile(scores, w, 0.50)
        assert q10 >= q50  # larger 1-alpha => higher quantile

    def test_uniform_weights_near_standard_quantile(self):
        sel = BandwidthSelector()
        rng = np.random.default_rng(2)
        scores = rng.exponential(100.0, 200)
        w = np.ones(200)
        wq = sel._weighted_quantile(scores, w, 0.10)
        std_q = float(np.quantile(scores, 0.90))
        assert abs(wq - std_q) / (std_q + 1) < 0.15

    def test_degenerate_weights_returns_inf_or_finite(self):
        sel = BandwidthSelector()
        scores = np.array([1.0, 2.0, 3.0])
        w = np.zeros(3)
        q = sel._weighted_quantile(scores, w, 0.10)
        assert q == np.inf or np.isfinite(q)


class TestSelect:
    def test_returns_bandwidth_cv_result(self, dummy_model, cal_data):
        sel = BandwidthSelector(
            candidates_km=[10.0, 20.0, 30.0],
            cv=3,
            grid_resolution=5,
        )
        scores = _make_scores(dummy_model, cal_data)
        result = sel.select(scores, cal_data["lat"], cal_data["lon"], alpha=0.10)
        assert isinstance(result, BandwidthCVResult)
        assert result.optimal_km in sel.candidates_km
        assert len(result.cv_scores) == len(sel.candidates_km)
        assert result.n_folds == 3

    def test_cv_scores_finite(self, dummy_model, cal_data):
        sel = BandwidthSelector(
            candidates_km=[10.0, 20.0],
            cv=3,
            grid_resolution=4,
        )
        scores = _make_scores(dummy_model, cal_data)
        result = sel.select(scores, cal_data["lat"], cal_data["lon"])
        # At least some CV scores should be finite
        finite_scores = [s for s in result.cv_scores if np.isfinite(s)]
        assert len(finite_scores) > 0

    def test_too_few_points_raises(self):
        sel = BandwidthSelector(cv=5)
        scores = np.ones(5)
        lat = np.linspace(51, 55, 5)
        lon = np.linspace(-1, 1, 5)
        with pytest.raises(ValueError, match="least"):
            sel.select(scores, lat, lon)

    def test_optimal_km_is_minimum_cv_score(self, dummy_model, cal_data):
        """Optimal bandwidth should correspond to the lowest CV score."""
        sel = BandwidthSelector(
            candidates_km=[5.0, 20.0, 50.0],
            cv=3,
            grid_resolution=4,
        )
        scores = _make_scores(dummy_model, cal_data)
        result = sel.select(scores, cal_data["lat"], cal_data["lon"])
        min_score_bw = result.candidates_km[int(np.argmin(result.cv_scores))]
        assert result.optimal_km == min_score_bw
