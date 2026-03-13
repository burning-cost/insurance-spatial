"""
Tests for predictor.py: SpatialConformalPredictor.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_spatial.conformal import SpatialConformalPredictor


class TestConstruction:
    def test_basic_construction(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model)
        assert scp.nonconformity == "pearson_tweedie"
        assert scp.tweedie_power == 1.5
        assert scp.spatial_kernel == "gaussian"
        assert not scp.is_calibrated_

    def test_invalid_model_raises(self):
        with pytest.raises(TypeError, match="predict"):
            SpatialConformalPredictor(model="not_a_model")

    def test_invalid_kernel_raises(self, dummy_model):
        with pytest.raises(ValueError, match="spatial_kernel"):
            SpatialConformalPredictor(model=dummy_model, spatial_kernel="square")

    def test_negative_bandwidth_raises(self, dummy_model):
        with pytest.raises(ValueError, match="positive"):
            SpatialConformalPredictor(model=dummy_model, bandwidth_km=-5.0)

    def test_all_kernels_accepted(self, dummy_model):
        for kernel in ("gaussian", "epanechnikov", "uniform"):
            scp = SpatialConformalPredictor(model=dummy_model, spatial_kernel=kernel)
            assert scp.spatial_kernel == kernel

    def test_all_nonconformity_scores_accepted(self, dummy_model, spread_model):
        for score in ("pearson_tweedie", "pearson", "absolute"):
            scp = SpatialConformalPredictor(model=dummy_model, nonconformity=score)
            assert scp.nonconformity == score

    def test_scaled_absolute_requires_spread_model(self, dummy_model):
        with pytest.raises(ValueError, match="spread_model"):
            SpatialConformalPredictor(
                model=dummy_model, nonconformity="scaled_absolute"
            )


class TestCalibrate:
    def test_basic_calibration(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(
            model=dummy_model, bandwidth_km=20.0
        )
        result = scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        assert scp.is_calibrated_
        assert result.n_calibration == len(cal_data["y"])
        assert result.bandwidth_km == 20.0
        assert not result.bandwidth_selected_by_cv
        assert result.score_name == "pearson_tweedie"
        assert result.score_mean > 0

    def test_calibration_stores_scores(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        scp.calibrate(cal_data["X"], cal_data["y"], lat=cal_data["lat"], lon=cal_data["lon"])
        assert scp.cal_scores_ is not None
        assert len(scp.cal_scores_) == len(cal_data["y"])
        assert np.all(scp.cal_scores_ >= 0)

    def test_calibration_stores_coords(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        scp.calibrate(cal_data["X"], cal_data["y"], lat=cal_data["lat"], lon=cal_data["lon"])
        np.testing.assert_array_equal(scp.cal_lat_, cal_data["lat"])
        np.testing.assert_array_equal(scp.cal_lon_, cal_data["lon"])

    def test_calibration_no_coords_raises(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        with pytest.raises(ValueError, match="lat.*lon.*postcodes"):
            scp.calibrate(cal_data["X"], cal_data["y"])

    def test_calibration_result_score_stats(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        result = scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        assert result.score_std >= 0
        assert 0 <= result.score_p50 <= result.score_p95

    def test_recalibration_overwrites(self, dummy_model, cal_data, test_data):
        """Calling calibrate() twice should update the stored scores."""
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        scp.calibrate(cal_data["X"], cal_data["y"], lat=cal_data["lat"], lon=cal_data["lon"])
        first_n = scp.cal_scores_.shape[0]

        # Calibrate on a smaller set
        n_small = 50
        scp.calibrate(
            cal_data["X"][:n_small], cal_data["y"][:n_small],
            lat=cal_data["lat"][:n_small], lon=cal_data["lon"][:n_small]
        )
        assert scp.cal_scores_.shape[0] == n_small
        assert scp.cal_scores_.shape[0] != first_n or first_n == n_small


class TestPredictInterval:
    def test_basic_prediction_shape(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        n = len(test_data["y"])
        assert result.lower.shape == (n,)
        assert result.upper.shape == (n,)
        assert result.point.shape == (n,)
        assert result.n_effective.shape == (n,)

    def test_lower_leq_upper(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        assert np.all(result.lower <= result.upper)

    def test_lower_nonnegative(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        assert np.all(result.lower >= 0)

    def test_alpha_stored(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"], alpha=0.05
        )
        assert result.alpha == 0.05

    def test_smaller_alpha_wider_intervals(self, calibrated_predictor, test_data):
        r90 = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"], alpha=0.10
        )
        r50 = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"], alpha=0.50
        )
        # 90% intervals (alpha=0.10) should be wider than 50% (alpha=0.50)
        assert np.mean(r90.width()) >= np.mean(r50.width())

    def test_predict_without_calibration_raises(self, dummy_model, test_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            scp.predict_interval(
                test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
            )

    def test_predict_no_coords_raises(self, calibrated_predictor, test_data):
        with pytest.raises(ValueError, match="lat.*lon.*postcodes"):
            calibrated_predictor.predict_interval(test_data["X"])

    def test_bandwidth_override(self, calibrated_predictor, test_data):
        r_narrow = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"],
            bandwidth_km=1.0
        )
        r_wide = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"],
            bandwidth_km=500.0
        )
        # Very wide bandwidth ~ uniform weighting ~ national quantile
        # Very narrow bandwidth may give different quantiles
        # Just check both produce valid intervals
        assert np.all(r_narrow.lower <= r_narrow.upper)
        assert np.all(r_wide.lower <= r_wide.upper)

    def test_n_effective_positive(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        assert np.all(result.n_effective > 0)

    def test_coverage_at_least_1_minus_alpha(self, dummy_model, cal_data, test_data):
        """
        Finite-sample conformal coverage guarantee: empirical coverage >= 1 - alpha.

        This is a soft test — conformal is a probabilistic guarantee over
        draws of the calibration/test split, so a single run may fall slightly below.
        We use a generous threshold.
        """
        # Use a large bandwidth so coverage is close to the flat conformal guarantee
        scp = SpatialConformalPredictor(
            model=dummy_model, bandwidth_km=10000.0  # effectively uniform
        )
        scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        alpha = 0.10
        result = scp.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"], alpha=alpha
        )
        covered = (test_data["y"] >= result.lower) & (test_data["y"] <= result.upper)
        empirical_cov = float(np.mean(covered))
        # Allow small tolerance below 1-alpha
        assert empirical_cov >= 1.0 - alpha - 0.10, (
            f"Empirical coverage {empirical_cov:.3f} too far below target {1-alpha}"
        )

    def test_interval_result_width_method(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        width = result.width()
        np.testing.assert_allclose(width, result.upper - result.lower, rtol=1e-10)

    def test_interval_result_relative_width(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        rw = result.relative_width()
        assert rw.shape == result.point.shape
        assert np.all(rw[result.point > 0] > 0)

    def test_epanechnikov_kernel(self, dummy_model, cal_data, test_data):
        scp = SpatialConformalPredictor(
            model=dummy_model, spatial_kernel="epanechnikov", bandwidth_km=30.0
        )
        scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        result = scp.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        assert np.all(result.lower <= result.upper)

    def test_uniform_kernel(self, dummy_model, cal_data, test_data):
        scp = SpatialConformalPredictor(
            model=dummy_model, spatial_kernel="uniform", bandwidth_km=200.0
        )
        scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        result = scp.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        assert np.all(result.lower <= result.upper)


class TestWeightedQuantile:
    def test_uniform_weights_matches_standard_quantile(self, dummy_model, cal_data):
        """With equal weights, weighted quantile should match standard."""
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=10000.0)
        scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        n = len(scp.cal_scores_)
        uniform_w = np.ones(n)
        alpha = 0.10
        wq = scp._weighted_quantile(scp.cal_scores_, uniform_w, alpha)
        std_q = float(np.quantile(scp.cal_scores_, 1.0 - alpha))
        # Should be close but not exact due to augmentation
        assert abs(wq - std_q) / (std_q + 1e-6) < 0.1

    def test_degenerate_weights_falls_back(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        zero_weights = np.zeros(len(scp.cal_scores_))
        q = scp._weighted_quantile(scp.cal_scores_, zero_weights, 0.10)
        assert np.isfinite(q) or q == np.inf
