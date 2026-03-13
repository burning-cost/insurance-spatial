"""
Tests for _types.py: result dataclasses.
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


class TestCalibrationResult:
    def test_construction(self):
        r = CalibrationResult(
            n_calibration=400,
            bandwidth_km=20.0,
            bandwidth_selected_by_cv=False,
            score_name="pearson_tweedie",
            score_mean=1.5,
            score_std=0.5,
            score_p50=1.4,
            score_p95=2.8,
        )
        assert r.n_calibration == 400
        assert r.bandwidth_km == 20.0
        assert r.score_name == "pearson_tweedie"


class TestIntervalResult:
    def _make_result(self, n=10):
        rng = np.random.default_rng(0)
        point = rng.gamma(2, 200, n)
        lower = point * 0.5
        upper = point * 1.5
        return IntervalResult(
            lower=lower,
            upper=upper,
            point=point,
            alpha=0.10,
            n_effective=np.full(n, 50.0),
            bandwidth_km=20.0,
        )

    def test_width_method(self):
        r = self._make_result()
        np.testing.assert_allclose(r.width(), r.upper - r.lower, rtol=1e-10)

    def test_relative_width_method(self):
        r = self._make_result()
        rw = r.relative_width()
        expected = r.width() / r.point
        np.testing.assert_allclose(rw, expected, rtol=1e-10)

    def test_relative_width_with_zero_point(self):
        r = IntervalResult(
            lower=np.array([0.0]),
            upper=np.array([100.0]),
            point=np.array([0.0]),
            alpha=0.10,
            n_effective=np.array([50.0]),
            bandwidth_km=20.0,
        )
        rw = r.relative_width()
        assert np.isnan(rw[0])


class TestCoverageResult:
    def test_coverage_gap(self):
        r = CoverageResult(
            alpha=0.10,
            target_coverage=0.90,
            marginal_coverage=0.87,
            macg=0.04,
            n_grid_cells=150,
            n_val=500,
        )
        assert r.coverage_gap() == pytest.approx(0.03, abs=1e-10)

    def test_coverage_gap_zero(self):
        r = CoverageResult(
            alpha=0.10,
            target_coverage=0.90,
            marginal_coverage=0.90,
            macg=0.0,
            n_grid_cells=100,
            n_val=300,
        )
        assert r.coverage_gap() == pytest.approx(0.0, abs=1e-10)


class TestBandwidthCVResult:
    def test_construction(self):
        r = BandwidthCVResult(
            optimal_km=20.0,
            candidates_km=[5.0, 10.0, 20.0, 30.0],
            cv_scores=[0.08, 0.05, 0.04, 0.045],
            metric="macg",
            n_folds=5,
        )
        assert r.optimal_km == 20.0
        assert r.metric == "macg"
        assert len(r.cv_scores) == 4
