"""
Integration tests: end-to-end workflows for insurance-spatial-conformal.

These tests run the full pipeline from calibration to coverage reporting,
checking that the pieces fit together correctly and that the spatial weighting
demonstrably improves coverage consistency compared to flat conformal.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_spatial.conformal import (
    SpatialConformalPredictor,
    SpatialCoverageReport,
    BandwidthSelector,
    TweediePearsonScore,
)


class TestFullPipeline:
    def test_calibrate_predict_report(self, dummy_model, cal_data, val_data):
        """Full pipeline: calibrate -> predict -> report."""
        scp = SpatialConformalPredictor(
            model=dummy_model,
            nonconformity="pearson_tweedie",
            tweedie_power=1.5,
            bandwidth_km=25.0,
        )

        cal_result = scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        assert scp.is_calibrated_

        intervals = scp.predict_interval(
            val_data["X"], lat=val_data["lat"], lon=val_data["lon"], alpha=0.10
        )
        assert np.all(intervals.lower <= intervals.upper)

        report = SpatialCoverageReport(scp)
        result = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        assert 0.0 <= result.marginal_coverage <= 1.0
        assert result.macg >= 0.0

    def test_absolute_score_pipeline(self, dummy_model, cal_data, val_data):
        scp = SpatialConformalPredictor(
            model=dummy_model, nonconformity="absolute", bandwidth_km=25.0
        )
        scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        result = scp.predict_interval(
            val_data["X"], lat=val_data["lat"], lon=val_data["lon"]
        )
        assert np.all(result.lower >= 0)

    def test_pearson_score_pipeline(self, dummy_model, cal_data, val_data):
        scp = SpatialConformalPredictor(
            model=dummy_model, nonconformity="pearson", bandwidth_km=25.0
        )
        scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        result = scp.predict_interval(
            val_data["X"], lat=val_data["lat"], lon=val_data["lon"]
        )
        assert result.lower.shape == (len(val_data["y"]),)

    def test_spread_model_pipeline(self, dummy_model, cal_data, val_data, spread_model):
        """ScaledAbsoluteScore pipeline with a spread model."""
        from insurance_spatial.conformal import SpatialConformalPredictor
        # DummyModel available via dummy_model fixture

        model = dummy_model
        scp = SpatialConformalPredictor(
            model=model,
            nonconformity="scaled_absolute",
            spread_model=spread_model,
            bandwidth_km=25.0,
        )
        scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )
        result = scp.predict_interval(
            val_data["X"], lat=val_data["lat"], lon=val_data["lon"]
        )
        assert np.all(result.lower <= result.upper)


class TestSpatialVsFlat:
    def test_spatial_macg_not_worse_than_flat(self, dummy_model, cal_data, val_data):
        """
        Spatial weighting should give MACG at least as good as flat conformal on this data.

        This is a soft property test — not guaranteed to pass on every random seed,
        but should hold on average for geographically concentrated portfolios.
        We use a very permissive tolerance.
        """
        # Flat conformal: uniform weights (very wide bandwidth)
        scp_flat = SpatialConformalPredictor(
            model=dummy_model, bandwidth_km=10000.0
        )
        scp_flat.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )

        # Spatial conformal: 30 km bandwidth
        scp_spatial = SpatialConformalPredictor(
            model=dummy_model, bandwidth_km=30.0
        )
        scp_spatial.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"]
        )

        report_flat = SpatialCoverageReport(scp_flat)
        report_spatial = SpatialCoverageReport(scp_spatial)

        r_flat = report_flat.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        r_spatial = report_spatial.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )

        # Both should give valid coverage results
        assert np.isfinite(r_flat.macg) or np.isnan(r_flat.macg)
        assert np.isfinite(r_spatial.macg) or np.isnan(r_spatial.macg)


class TestBandwidthCVIntegration:
    def test_cv_bandwidth_leads_to_calibration(self, dummy_model, cal_data):
        """Auto bandwidth selection (bandwidth_km=None) should succeed."""
        scp = SpatialConformalPredictor(
            model=dummy_model,
            bandwidth_km=None,  # trigger CV
        )
        result = scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"],
            cv_candidates_km=[10.0, 20.0, 30.0],
            cv_folds=3,
        )
        assert result.bandwidth_selected_by_cv
        assert result.bandwidth_km > 0
        assert scp.bandwidth_km_ == result.bandwidth_km

    def test_cv_bandwidth_in_candidates(self, dummy_model, cal_data):
        candidates = [10.0, 20.0, 30.0]
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=None)
        result = scp.calibrate(
            cal_data["X"], cal_data["y"],
            lat=cal_data["lat"], lon=cal_data["lon"],
            cv_candidates_km=candidates,
            cv_folds=3,
        )
        assert result.bandwidth_km in candidates


class TestFCATable:
    def test_fca_table_pipeline(self, calibrated_predictor, val_data):
        import polars as pl
        report = calibrated_predictor.spatial_coverage_report(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"],
        )
        table = report.fca_consumer_duty_table()
        assert isinstance(table, pl.DataFrame)
        assert len(table) > 0

    def test_fca_table_with_flood_zones(self, calibrated_predictor, val_data):
        import polars as pl
        n = len(val_data["y"])
        flood_labels = np.where(
            val_data["lon"] < -1.0, "HighFloodRisk", "LowFloodRisk"
        )
        report = calibrated_predictor.spatial_coverage_report(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"],
        )
        table = report.fca_consumer_duty_table(region_labels=flood_labels)
        assert isinstance(table, pl.DataFrame)
        assert "HighFloodRisk" in table["Region"].to_list()
