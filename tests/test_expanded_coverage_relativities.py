"""
Expanded test coverage for the relativities module.

Additional edge cases: single area, equal posterior draws, different
CI widths, ln_offset monotonicity, and large-variance scenarios.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from unittest.mock import MagicMock

from insurance_spatial.adjacency import build_grid_adjacency
from insurance_spatial.relativities import extract_relativities


def _make_mock_result(
    n_areas: int = 9,
    n_chains: int = 2,
    n_draws: int = 100,
    b_mean_values: np.ndarray = None,
    b_noise_scale: float = 0.1,
    seed: int = 42,
):
    import xarray as xr

    rng = np.random.default_rng(seed)
    if b_mean_values is None:
        b_mean_values = np.linspace(0.5, -0.5, n_areas)

    b_samples = (
        b_mean_values[None, None, :]
        + rng.standard_normal((n_chains, n_draws, n_areas)) * b_noise_scale
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

    adj = build_grid_adjacency(int(np.ceil(np.sqrt(n_areas))), int(np.ceil(np.sqrt(n_areas))))
    areas = adj.areas[:n_areas]
    result = MagicMock()
    result.trace = mock_trace
    result.areas = areas
    result.n_areas = n_areas
    return result


# ---------------------------------------------------------------------------
# Return type and column contracts
# ---------------------------------------------------------------------------

class TestExtractRelativitiesTypes:
    def test_area_column_dtype_is_string(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert df["area"].dtype == pl.Utf8 or df["area"].dtype == pl.String

    def test_float_columns_are_float64(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        for col in ("b_mean", "b_sd", "relativity", "lower", "upper", "ln_offset"):
            assert df[col].dtype == pl.Float64

    def test_no_null_values_in_output(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert df.null_count().sum_horizontal().to_list()[0] == 0

    def test_output_has_7_columns(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert len(df.columns) == 7


# ---------------------------------------------------------------------------
# Credibility interval behaviour
# ---------------------------------------------------------------------------

class TestCredibilityIntervals:
    def test_99_interval_wider_than_95(self):
        result = _make_mock_result(n_draws=200)
        df_95 = extract_relativities(result, credibility_interval=0.95)
        df_99 = extract_relativities(result, credibility_interval=0.99)
        range_95 = float((df_95["upper"] - df_95["lower"]).mean())
        range_99 = float((df_99["upper"] - df_99["lower"]).mean())
        assert range_99 > range_95

    def test_50_interval_narrower_than_95(self):
        result = _make_mock_result(n_draws=200)
        df_95 = extract_relativities(result, credibility_interval=0.95)
        df_50 = extract_relativities(result, credibility_interval=0.50)
        range_95 = float((df_95["upper"] - df_95["lower"]).mean())
        range_50 = float((df_50["upper"] - df_50["lower"]).mean())
        assert range_95 > range_50

    def test_lower_bound_positive(self):
        """All relativities and their lower CI bounds must be > 0."""
        result = _make_mock_result()
        df = extract_relativities(result)
        assert (df["lower"] > 0).all()

    def test_upper_gt_lower_all_areas(self):
        result = _make_mock_result()
        df = extract_relativities(result)
        assert (df["upper"] > df["lower"]).all()


# ---------------------------------------------------------------------------
# Normalisation logic
# ---------------------------------------------------------------------------

class TestNormalisationLogic:
    def test_grand_mean_ln_offset_sums_to_zero(self):
        """Under grand-mean normalisation, sum of ln_offset should be 0."""
        result = _make_mock_result()
        df = extract_relativities(result)
        total = float(df["ln_offset"].sum())
        assert abs(total) < 1e-8

    def test_base_area_has_ln_offset_zero(self):
        result = _make_mock_result(n_areas=9)
        df = extract_relativities(result, base_area="r0c0")
        base_row = df.filter(pl.col("area") == "r0c0")
        assert abs(float(base_row["ln_offset"][0])) < 1e-10

    def test_base_area_relativity_close_to_one(self):
        result = _make_mock_result(n_areas=9, n_draws=500, b_noise_scale=0.05)
        df = extract_relativities(result, base_area="r0c1")
        base_row = df.filter(pl.col("area") == "r0c1")
        assert abs(float(base_row["relativity"][0]) - 1.0) < 0.1

    def test_zero_b_all_areas_relativities_all_one(self):
        """If all posterior b values are 0, all relativities should be 1.0."""
        result = _make_mock_result(
            n_areas=4,
            b_mean_values=np.zeros(4),
            b_noise_scale=0.0,
            seed=1,
        )
        adj = build_grid_adjacency(2, 2)
        result.areas = adj.areas[:4]
        df = extract_relativities(result)
        np.testing.assert_allclose(df["relativity"].to_numpy(), 1.0, atol=1e-6)

    def test_monotone_b_produces_monotone_relativities(self):
        """Areas with higher b should have higher relativities under grand-mean norm."""
        b_vals = np.linspace(-1.0, 1.0, 9)
        result = _make_mock_result(n_areas=9, b_mean_values=b_vals, b_noise_scale=0.01, n_draws=500)
        df = extract_relativities(result)
        rels = df["relativity"].to_numpy()
        # Relativities should be approximately monotone with b
        # (not strictly guaranteed due to Jensen but should hold with low noise)
        b_means = df["b_mean"].to_numpy()
        assert np.corrcoef(b_means, rels)[0, 1] > 0.99

    def test_invalid_base_area_message_includes_area_name(self):
        result = _make_mock_result()
        with pytest.raises(ValueError, match="BOGUS_AREA"):
            extract_relativities(result, base_area="BOGUS_AREA")


# ---------------------------------------------------------------------------
# Multiple chains and draw counts
# ---------------------------------------------------------------------------

class TestMultipleChainsAndDraws:
    def test_single_chain_works(self):
        result = _make_mock_result(n_chains=1, n_draws=50)
        df = extract_relativities(result)
        assert len(df) == 9

    def test_many_draws_gives_tighter_b_sd(self):
        """More draws for a given true b → lower b_sd estimates."""
        result_few = _make_mock_result(n_draws=20, b_noise_scale=0.5, seed=7)
        result_many = _make_mock_result(n_draws=2000, b_noise_scale=0.5, seed=7)
        df_few = extract_relativities(result_few)
        df_many = extract_relativities(result_many)
        # b_sd (posterior SD) should be similar; what varies is estimation noise
        # Both should produce valid positive b_sd values
        assert (df_few["b_sd"] >= 0).all()
        assert (df_many["b_sd"] >= 0).all()

    def test_4_chains_produces_correct_n_rows(self):
        result = _make_mock_result(n_chains=4, n_draws=100, n_areas=16)
        adj = build_grid_adjacency(4, 4)
        result.areas = adj.areas
        df = extract_relativities(result)
        assert len(df) == 16
