"""
Expanded test coverage for conformal/predictor.py.

Additional edge cases: exposure handling, _predict_model internals,
_resolve_coords, predict_interval with various configurations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_spatial.conformal import SpatialConformalPredictor
from insurance_spatial.conformal._types import CalibrationResult, IntervalResult


# ---------------------------------------------------------------------------
# Fixtures reuse conftest.py from conformal/
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Construction edge cases
# ---------------------------------------------------------------------------

class TestConstructionEdgeCases:
    def test_bandwidth_zero_raises(self, dummy_model):
        with pytest.raises(ValueError, match="positive"):
            SpatialConformalPredictor(model=dummy_model, bandwidth_km=0.0)

    def test_bandwidth_very_small_positive_allowed(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=1e-6)
        assert scp.bandwidth_km == pytest.approx(1e-6)

    def test_n_eff_min_stored(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model, n_eff_min=50, bandwidth_km=20.0)
        assert scp.n_eff_min == 50

    def test_tweedie_power_stored(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model, tweedie_power=2.0, bandwidth_km=20.0)
        assert scp.tweedie_power == pytest.approx(2.0)

    def test_is_calibrated_initially_false(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        assert scp.is_calibrated_ is False

    def test_cal_scores_initially_none(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        assert scp.cal_scores_ is None


# ---------------------------------------------------------------------------
# _predict_model: exposure handling
# ---------------------------------------------------------------------------

class TestPredictModelExposure:
    def test_exposure_multiplied_into_predictions(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        X = cal_data["X"][:10]
        exposure = np.full(10, 2.0)

        yhat_no_exp = scp._predict_model(X, None)
        yhat_with_exp = scp._predict_model(X, exposure)

        np.testing.assert_allclose(yhat_with_exp, yhat_no_exp * 2.0, rtol=1e-10)

    def test_exposure_wrong_shape_raises(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        X = cal_data["X"][:10]
        bad_exposure = np.ones(5)  # wrong size

        with pytest.raises(ValueError, match="exposure shape"):
            scp._predict_model(X, bad_exposure)

    def test_no_exposure_returns_raw_predictions(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        X = cal_data["X"][:5]
        yhat = scp._predict_model(X, None)
        expected = dummy_model.predict(X)
        np.testing.assert_allclose(yhat, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# _resolve_coords
# ---------------------------------------------------------------------------

class TestResolveCoords:
    def test_lat_lon_provided(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        lat = np.array([51.5, 52.0])
        lon = np.array([-0.1, -1.0])
        out_lat, out_lon = scp._resolve_coords(lat, lon, None, "test")
        np.testing.assert_array_equal(out_lat, lat)
        np.testing.assert_array_equal(out_lon, lon)

    def test_neither_provided_raises(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        with pytest.raises(ValueError, match="lat.*lon.*postcodes"):
            scp._resolve_coords(None, None, None, "test")

    def test_lat_without_lon_raises(self, dummy_model):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        lat = np.array([51.5])
        # lon is None; lat provided but lon missing → falls through to ValueError
        with pytest.raises(ValueError):
            scp._resolve_coords(lat, None, None, "test")


# ---------------------------------------------------------------------------
# Calibration with exposure
# ---------------------------------------------------------------------------

class TestCalibrateWithExposure:
    def test_calibration_with_exposure(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        n = len(cal_data["y"])
        exposure = np.ones(n) * 2.0

        result = scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"],
            exposure=exposure,
        )
        assert result.n_calibration == n
        assert scp.is_calibrated_

    def test_scores_differ_with_and_without_exposure(self, dummy_model, cal_data):
        scp1 = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        scp1.calibrate(cal_data["X"], cal_data["y"], lat=cal_data["lat"], lon=cal_data["lon"])

        scp2 = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        exposure = np.ones(len(cal_data["y"])) * 3.0
        scp2.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"],
            exposure=exposure,
        )

        # Scores should differ because predictions are scaled by exposure
        assert not np.allclose(scp1.cal_scores_, scp2.cal_scores_)


# ---------------------------------------------------------------------------
# predict_interval additional cases
# ---------------------------------------------------------------------------

class TestPredictIntervalAdditional:
    def test_predict_with_exposure(self, calibrated_predictor, test_data):
        exposure = np.ones(len(test_data["y"])) * 2.0
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"],
            exposure=exposure,
        )
        # Predictions with 2x exposure should be ~2x wider intervals
        result_no_exp = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"],
        )
        # Check intervals exist and are valid
        assert np.all(result.lower <= result.upper)
        assert np.all(result_no_exp.lower <= result_no_exp.upper)

    def test_bandwidth_km_stored_in_result(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        assert result.bandwidth_km == pytest.approx(calibrated_predictor.bandwidth_km_)

    def test_bandwidth_override_stored_in_result(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"],
            bandwidth_km=99.0,
        )
        assert result.bandwidth_km == pytest.approx(99.0)

    def test_low_n_eff_warns(self, dummy_model, cal_data, test_data):
        """Very narrow bandwidth should trigger a low n_eff warning for remote test points."""
        scp = SpatialConformalPredictor(
            model=dummy_model,
            bandwidth_km=0.001,  # almost no calibration data nearby
            n_eff_min=30,
        )
        scp.calibrate(cal_data["X"], cal_data["y"], lat=cal_data["lat"], lon=cal_data["lon"])

        with pytest.warns(UserWarning, match="effective calibration N"):
            scp.predict_interval(
                test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
            )

    def test_single_test_point(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"][:1],
            lat=test_data["lat"][:1],
            lon=test_data["lon"][:1],
        )
        assert result.lower.shape == (1,)
        assert result.upper.shape == (1,)

    def test_result_is_interval_result_type(self, calibrated_predictor, test_data):
        result = calibrated_predictor.predict_interval(
            test_data["X"], lat=test_data["lat"], lon=test_data["lon"]
        )
        assert isinstance(result, IntervalResult)

    def test_calibration_result_is_calibration_result_type(self, dummy_model, cal_data):
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        result = scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"],
        )
        assert isinstance(result, CalibrationResult)
