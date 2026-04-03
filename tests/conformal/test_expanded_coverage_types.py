"""
Expanded test coverage for conformal/_types.py.

Tests additional methods and edge cases on result dataclasses.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_spatial.conformal._types import (
    CalibrationResult,
    IntervalResult,
    CoverageResult,
    BandwidthCVResult,
)


class TestCalibrationResultAdditional:
    def test_bandwidth_selected_by_cv_false(self):
        r = CalibrationResult(
            n_calibration=100,
            bandwidth_km=15.0,
            bandwidth_selected_by_cv=False,
            score_name="absolute",
            score_mean=50.0,
            score_std=20.0,
            score_p50=45.0,
            score_p95=90.0,
        )
        assert r.bandwidth_selected_by_cv is False
        assert r.score_name == "absolute"

    def test_bandwidth_selected_by_cv_true(self):
        r = CalibrationResult(
            n_calibration=300,
            bandwidth_km=25.0,
            bandwidth_selected_by_cv=True,
            score_name="pearson",
            score_mean=1.2,
            score_std=0.4,
            score_p50=1.1,
            score_p95=2.0,
        )
        assert r.bandwidth_selected_by_cv is True

    def test_all_score_stats_accessible(self):
        r = CalibrationResult(
            n_calibration=200,
            bandwidth_km=10.0,
            bandwidth_selected_by_cv=False,
            score_name="pearson_tweedie",
            score_mean=0.9,
            score_std=0.3,
            score_p50=0.85,
            score_p95=1.6,
        )
        assert r.score_mean == pytest.approx(0.9)
        assert r.score_std == pytest.approx(0.3)
        assert r.score_p50 == pytest.approx(0.85)
        assert r.score_p95 == pytest.approx(1.6)


class TestIntervalResultAdditional:
    def _make(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        point = rng.gamma(2, 300, n)
        lower = point * 0.6
        upper = point * 1.8
        return IntervalResult(
            lower=lower,
            upper=upper,
            point=point,
            alpha=0.10,
            n_effective=np.full(n, 40.0),
            bandwidth_km=25.0,
        )

    def test_alpha_stored(self):
        r = self._make()
        assert r.alpha == pytest.approx(0.10)

    def test_bandwidth_km_stored(self):
        r = self._make()
        assert r.bandwidth_km == pytest.approx(25.0)

    def test_n_effective_shape(self):
        r = self._make(n=15)
        assert r.n_effective.shape == (15,)

    def test_width_always_non_negative(self):
        r = self._make()
        assert np.all(r.width() >= 0)

    def test_relative_width_all_positive_when_point_positive(self):
        r = self._make()
        rw = r.relative_width()
        assert np.all(rw[r.point > 0] > 0)

    def test_relative_width_nan_when_point_zero(self):
        r = IntervalResult(
            lower=np.array([0.0, 10.0]),
            upper=np.array([100.0, 200.0]),
            point=np.array([0.0, 100.0]),
            alpha=0.10,
            n_effective=np.array([30.0, 30.0]),
            bandwidth_km=20.0,
        )
        rw = r.relative_width()
        assert np.isnan(rw[0])
        assert not np.isnan(rw[1])

    def test_width_equals_upper_minus_lower(self):
        r = self._make(n=50)
        np.testing.assert_allclose(r.width(), r.upper - r.lower, rtol=1e-12)

    def test_point_inside_interval(self):
        """Point prediction should be within [lower, upper]."""
        r = self._make()
        assert np.all(r.lower <= r.point)
        assert np.all(r.point <= r.upper)


class TestCoverageResultAdditional:
    def test_coverage_gap_positive_when_undercovered(self):
        r = CoverageResult(
            alpha=0.10,
            target_coverage=0.90,
            marginal_coverage=0.82,
            macg=0.07,
            n_grid_cells=80,
            n_val=400,
        )
        assert r.coverage_gap() > 0

    def test_coverage_gap_negative_when_overcovered(self):
        r = CoverageResult(
            alpha=0.05,
            target_coverage=0.95,
            marginal_coverage=0.97,
            macg=0.02,
            n_grid_cells=120,
            n_val=600,
        )
        assert r.coverage_gap() < 0

    def test_optional_fields_default_none(self):
        r = CoverageResult(
            alpha=0.10,
            target_coverage=0.90,
            marginal_coverage=0.88,
            macg=0.05,
            n_grid_cells=100,
            n_val=500,
        )
        assert r.coverage_by_cell is None
        assert r.cell_centres_lat is None
        assert r.cell_centres_lon is None
        assert r.coverage_by_region is None

    def test_optional_fields_set(self):
        cells = np.array([0.88, 0.91, 0.93])
        r = CoverageResult(
            alpha=0.10,
            target_coverage=0.90,
            marginal_coverage=0.90,
            macg=0.02,
            n_grid_cells=3,
            n_val=200,
            coverage_by_cell=cells,
            cell_centres_lat=np.array([51.5, 53.0, 55.0]),
            cell_centres_lon=np.array([-0.1, -2.0, -4.0]),
        )
        assert r.coverage_by_cell is not None
        assert r.cell_centres_lat is not None

    def test_target_coverage_is_one_minus_alpha(self):
        """A correctly constructed result should have target = 1 - alpha."""
        alpha = 0.05
        r = CoverageResult(
            alpha=alpha,
            target_coverage=1.0 - alpha,
            marginal_coverage=0.94,
            macg=0.02,
            n_grid_cells=50,
            n_val=300,
        )
        assert r.target_coverage == pytest.approx(1.0 - alpha)


class TestBandwidthCVResultAdditional:
    def test_n_folds_stored(self):
        r = BandwidthCVResult(
            optimal_km=15.0,
            candidates_km=[5.0, 15.0, 30.0],
            cv_scores=[0.09, 0.04, 0.06],
            metric="macg",
            n_folds=4,
        )
        assert r.n_folds == 4

    def test_optimal_km_in_candidates(self):
        candidates = [5.0, 10.0, 20.0, 30.0]
        r = BandwidthCVResult(
            optimal_km=20.0,
            candidates_km=candidates,
            cv_scores=[0.08, 0.05, 0.04, 0.045],
            metric="macg",
            n_folds=5,
        )
        assert r.optimal_km in r.candidates_km

    def test_cv_scores_length_matches_candidates(self):
        candidates = [2.0, 5.0, 10.0]
        scores = [0.1, 0.06, 0.08]
        r = BandwidthCVResult(
            optimal_km=5.0,
            candidates_km=candidates,
            cv_scores=scores,
            metric="macg",
            n_folds=5,
        )
        assert len(r.cv_scores) == len(r.candidates_km)

    def test_optimal_km_mutable(self):
        """BandwidthCVResult.optimal_km must be mutable (used by select_with_n_eff_floor)."""
        r = BandwidthCVResult(
            optimal_km=10.0,
            candidates_km=[10.0, 20.0],
            cv_scores=[0.05, 0.06],
            metric="macg",
            n_folds=5,
        )
        r.optimal_km = 20.0
        assert r.optimal_km == 20.0
