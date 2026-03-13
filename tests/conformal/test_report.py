"""
Tests for report.py: SpatialCoverageReport.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_spatial.conformal import SpatialCoverageReport
from insurance_spatial.conformal._types import CoverageResult


class TestSpatialCoverageReportConstruction:
    def test_requires_calibrated_predictor(self, dummy_model):
        from insurance_spatial.conformal import SpatialConformalPredictor
        scp = SpatialConformalPredictor(model=dummy_model, bandwidth_km=20.0)
        with pytest.raises(RuntimeError, match="calibrated"):
            SpatialCoverageReport(scp)

    def test_accepts_calibrated_predictor(self, calibrated_predictor):
        report = SpatialCoverageReport(calibrated_predictor)
        assert report.predictor is calibrated_predictor


class TestEvaluate:
    def test_returns_coverage_result(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        result = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"],
            alpha=0.10,
        )
        assert isinstance(result, CoverageResult)

    def test_marginal_coverage_in_range(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        result = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        assert 0.0 <= result.marginal_coverage <= 1.0

    def test_macg_non_negative(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        result = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        assert result.macg >= 0.0

    def test_n_val_correct(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        result = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        assert result.n_val == len(val_data["y"])

    def test_target_coverage_set(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        result = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"],
            alpha=0.05
        )
        assert result.target_coverage == pytest.approx(0.95)
        assert result.alpha == pytest.approx(0.05)

    def test_coverage_gap_method(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        result = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        expected_gap = result.target_coverage - result.marginal_coverage
        assert result.coverage_gap() == pytest.approx(expected_gap, abs=1e-10)

    def test_no_coords_raises(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        with pytest.raises(ValueError, match="lat.*lon.*postcodes"):
            report.evaluate(val_data["X"], val_data["y"])

    def test_grid_resolution_affects_n_cells(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        r5 = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"],
            grid_resolution=5
        )
        report2 = SpatialCoverageReport(calibrated_predictor)
        r20 = report2.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"],
            grid_resolution=20
        )
        # More grid cells at higher resolution (subject to min 3 obs per cell)
        assert r20.n_grid_cells >= r5.n_grid_cells

    def test_cell_coverage_shape(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        result = report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"],
            grid_resolution=10
        )
        if result.coverage_by_cell is not None:
            assert result.coverage_by_cell.shape == (result.n_grid_cells,)
            assert np.all(result.coverage_by_cell >= 0)
            assert np.all(result.coverage_by_cell <= 1)


class TestCoverageMap:
    def test_returns_figure(self, calibrated_predictor, val_data):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")

        report = SpatialCoverageReport(calibrated_predictor)
        report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        import matplotlib.figure
        fig = report.coverage_map(resolution=5)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_without_evaluate_raises(self, calibrated_predictor):
        report = SpatialCoverageReport(calibrated_predictor)
        with pytest.raises(RuntimeError, match="evaluate"):
            report.coverage_map()


class TestFCAConsumerDutyTable:
    def test_returns_polars_dataframe(self, calibrated_predictor, val_data):
        import polars as pl
        report = SpatialCoverageReport(calibrated_predictor)
        report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        table = report.fca_consumer_duty_table()
        assert isinstance(table, pl.DataFrame)

    def test_required_columns_present(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        table = report.fca_consumer_duty_table()
        for col in ("n_policies", "coverage", "target_coverage", "coverage_gap", "flag"):
            assert col in table.columns

    def test_flag_values_valid(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        table = report.fca_consumer_duty_table()
        valid_flags = {"OK", "MONITOR", "REVIEW"}
        flags = set(table["flag"].to_list())
        assert flags.issubset(valid_flags)

    def test_custom_region_labels(self, calibrated_predictor, val_data):
        import polars as pl
        n = len(val_data["y"])
        region_labels = np.array(["FloodZone" if i % 5 == 0 else "Standard" for i in range(n)])
        report = SpatialCoverageReport(calibrated_predictor)
        report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        table = report.fca_consumer_duty_table(region_labels=region_labels)
        assert isinstance(table, pl.DataFrame)
        assert "FloodZone" in table["Region"].to_list()

    def test_without_evaluate_raises(self, calibrated_predictor):
        report = SpatialCoverageReport(calibrated_predictor)
        with pytest.raises(RuntimeError, match="evaluate"):
            report.fca_consumer_duty_table()


class TestMacgByRegion:
    def test_returns_polars_dataframe(self, calibrated_predictor, val_data):
        import polars as pl
        n = len(val_data["y"])
        regions = np.array(["North" if val_data["lat"][i] > 53 else "South" for i in range(n)])
        report = SpatialCoverageReport(calibrated_predictor)
        report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        df = report.macg_by_region(regions)
        assert isinstance(df, pl.DataFrame)
        assert "macg" in df.columns

    def test_without_evaluate_raises(self, calibrated_predictor, val_data):
        n = len(val_data["y"])
        regions = np.array(["A"] * n)
        report = SpatialCoverageReport(calibrated_predictor)
        with pytest.raises(RuntimeError, match="evaluate"):
            report.macg_by_region(regions)


class TestSummary:
    def test_returns_string(self, calibrated_predictor, val_data):
        report = SpatialCoverageReport(calibrated_predictor)
        report.evaluate(
            val_data["X"], val_data["y"],
            lat=val_data["lat"], lon=val_data["lon"]
        )
        s = report.summary()
        assert isinstance(s, str)
        assert "MACG" in s
        assert "coverage" in s.lower()

    def test_without_evaluate_raises(self, calibrated_predictor):
        report = SpatialCoverageReport(calibrated_predictor)
        with pytest.raises(RuntimeError, match="evaluate"):
            report.summary()
