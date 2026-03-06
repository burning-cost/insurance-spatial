"""
Tests for the relativities module.

These tests construct a mock BYM2Result using synthetic trace data,
without requiring a real PyMC model fit.  This keeps the unit tests fast
and runnable locally.

Integration tests that call model.fit() are in test_model_integration.py
and are marked for Databricks execution.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from unittest.mock import MagicMock

from insurance_spatial.adjacency import build_grid_adjacency
from insurance_spatial.relativities import extract_relativities


def _make_mock_result(n_areas: int = 9, n_chains: int = 2, n_draws: int = 100):
    """
    Build a minimal mock BYM2Result with synthetic posterior data.
    Uses xarray DataArray format to match what PyMC/ArviZ produce.
    """
    import xarray as xr

    rng = np.random.default_rng(42)
    # Synthetic b posterior: shape (chains, draws, n_areas)
    # Give area 0 a higher b (riskier) and area 8 a lower b
    true_b = np.linspace(0.5, -0.5, n_areas)
    b_samples = (
        true_b[None, None, :]  # (1, 1, N)
        + rng.standard_normal((n_chains, n_draws, n_areas)) * 0.1
    )

    posterior_data = xr.Dataset(
        {
            "b": xr.DataArray(
                b_samples,
                dims=["chain", "draw", "b_dim_0"],
            )
        }
    )

    mock_trace = MagicMock()
    mock_trace.posterior = posterior_data

    adj = build_grid_adjacency(3, 3)
    result = MagicMock()
    result.trace = mock_trace
    result.areas = adj.areas  # ['r0c0', 'r0c1', ..., 'r2c2']
    result.adjacency = adj
    result.n_areas = n_areas

    return result


class TestExtractRelativities:
    def test_returns_polars_dataframe(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert isinstance(df, pl.DataFrame)

    def test_expected_columns(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert "area" in df.columns
        assert "relativity" in df.columns
        assert "lower" in df.columns
        assert "upper" in df.columns
        assert "b_mean" in df.columns
        assert "b_sd" in df.columns
        assert "ln_offset" in df.columns

    def test_n_rows_matches_n_areas(self):
        result = _make_mock_result(n_areas=9)
        df = extract_relativities(result)
        assert len(df) == 9

    def test_relativities_positive(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert (df["relativity"] > 0).all()

    def test_lower_less_than_upper(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert (df["lower"] <= df["relativity"]).all()
        assert (df["relativity"] <= df["upper"]).all()

    def test_grand_mean_normalisation(self):
        """With no base_area, geometric mean of relativities should be ~1."""
        result = _make_mock_result()
        df = extract_relativities(result)
        geo_mean = np.exp(np.log(df["relativity"].to_numpy()).mean())
        assert abs(geo_mean - 1.0) < 0.05  # within 5%

    def test_base_area_normalisation(self):
        """Specified base_area should get relativity of exactly 1.0 in expectation."""
        result = _make_mock_result()
        df = extract_relativities(result, base_area="r1c1")
        base_row = df.filter(pl.col("area") == "r1c1")
        # Should be close to 1.0 (not exact because we use posterior mean not sample)
        assert abs(float(base_row["relativity"][0]) - 1.0) < 0.01

    def test_invalid_base_area_raises(self):
        result = _make_mock_result()
        with pytest.raises(ValueError, match="not found"):
            extract_relativities(result, base_area="INVALID_SECTOR")

    def test_ln_offset_is_log_of_relativity(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        computed = np.log(df["relativity"].to_numpy())
        stored = df["ln_offset"].to_numpy()
        np.testing.assert_allclose(computed, stored, atol=1e-10)

    def test_ordering_preserved(self):
        """Areas in output should be in same order as result.areas."""
        result = _make_mock_result(n_areas=9)
        df = extract_relativities(result)
        assert df["area"].to_list() == result.areas

    def test_wider_interval_gives_wider_bounds(self):
        result = _make_mock_result()
        df_90 = extract_relativities(result, credibility_interval=0.90)
        df_95 = extract_relativities(result, credibility_interval=0.95)
        # 95% CI should be wider than 90% CI
        range_90 = (df_90["upper"] - df_90["lower"]).mean()
        range_95 = (df_95["upper"] - df_95["lower"]).mean()
        assert float(range_95) > float(range_90)

    def test_high_b_area_has_relativity_above_1(self):
        """Area with highest b should have relativity > 1 under grand-mean normalisation."""
        result = _make_mock_result()
        df = extract_relativities(result)
        # r0c0 has highest b in our synthetic data (true_b[0] = 0.5)
        top_row = df.filter(pl.col("area") == "r0c0")
        assert float(top_row["relativity"][0]) > 1.0

    def test_b_sd_positive(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert (df["b_sd"] >= 0).all()
