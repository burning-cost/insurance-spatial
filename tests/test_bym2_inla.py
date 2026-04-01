"""
Tests for the BYM2InlaModel and BYM2InlaResult classes.

Test strategy
-------------
pyINLA is an optional dependency and is not installable on ARM64 Linux (the
Raspberry Pi dev machine).  Most tests here use a mock/stub of pyINLA's result
object to test all the logic that lives in our code:

  - Input validation in BYM2InlaModel.fit()
  - _write_adj_file() adjacency serialisation
  - _parse_result() mapping from pyINLA output to BYM2InlaResult
  - BYM2InlaResult.territory_relativities() calculations
  - BYM2InlaResult.diagnostics() structure

The full integration test (TestBYM2InlaIntegration) actually calls
pyinla.inla().  It is skipped when pyINLA is not installed.

When pyINLA is available on Databricks (x86_64 Linux), run the full test suite:
    databricks jobs run-now --job-id <job-id>

Design note: we mock at the pyinla.inla() boundary (in BYM2InlaModel.fit()),
not deeper.  This gives us confidence that our DataFrame construction,
.adj file generation, and result parsing are correct without needing the
actual INLA binary.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from insurance_spatial.adjacency import build_grid_adjacency, AdjacencyMatrix
from insurance_spatial.bym2_inla import (
    BYM2InlaModel,
    BYM2InlaResult,
    InlaDiagnostics,
    _write_adj_file,
    _parse_summary_df,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid_3x3() -> AdjacencyMatrix:
    return build_grid_adjacency(3, 3, connectivity="rook")


@pytest.fixture(scope="module")
def grid_5x5() -> AdjacencyMatrix:
    return build_grid_adjacency(5, 5, connectivity="rook")


@pytest.fixture(scope="module")
def synthetic_data_3x3(grid_3x3):
    """N=9 synthetic Poisson data on 3x3 grid."""
    rng = np.random.default_rng(42)
    N = grid_3x3.n  # 9
    exposure = rng.uniform(50.0, 300.0, size=N)
    true_log_rate = 0.3 - 0.1 * np.arange(N) * 0.05
    mu = exposure * np.exp(true_log_rate)
    claims = rng.poisson(mu)
    return claims, exposure


@pytest.fixture(scope="module")
def synthetic_data_5x5(grid_5x5):
    """N=25 synthetic Poisson data on 5x5 grid."""
    rng = np.random.default_rng(2024)
    N = grid_5x5.n  # 25
    exposure = rng.uniform(50.0, 500.0, size=N)
    row_idx = np.array([i // 5 for i in range(N)])
    true_log_rate = 0.5 - 0.2 * row_idx + 0.1 * rng.standard_normal(N)
    mu = exposure * np.exp(true_log_rate)
    claims = rng.poisson(mu)
    return claims, exposure


def _make_mock_inla_result(N: int, n_fixed: int = 0) -> MagicMock:
    """
    Build a MagicMock that looks like a pyINLA result object.

    Mimics the pyINLA v0.2.0 result structure:
      result.summary.random['area_id']  - DataFrame with mean, sd per area
      result.summary.fixed              - DataFrame with fixed effect posteriors
      result.summary.hyperpar           - DataFrame with hyperparameter posteriors
      result.dic                        - float
      result.waic                       - float
      result.marginals                  - dict
    """
    import pandas as pd

    rng = np.random.default_rng(99)

    # Random effects: N areas
    b_mean_vals = rng.standard_normal(N) * 0.3
    b_sd_vals = np.abs(rng.standard_normal(N)) * 0.1 + 0.05
    random_df = pd.DataFrame(
        {
            "mean": b_mean_vals,
            "sd": b_sd_vals,
            "0.025quant": b_mean_vals - 1.96 * b_sd_vals,
            "0.975quant": b_mean_vals + 1.96 * b_sd_vals,
        }
    )

    # Fixed effects
    fixed_rows = {"mean": [0.1] * (n_fixed + 1),  # alpha + covariates
                  "sd": [0.05] * (n_fixed + 1),
                  "0.025quant": [0.0] * (n_fixed + 1),
                  "0.975quant": [0.2] * (n_fixed + 1)}
    fixed_index = ["alpha"] + [f"x{i}" for i in range(n_fixed)]
    fixed_df = pd.DataFrame(fixed_rows, index=fixed_index)

    # Hyperparameters
    hyper_rows = {"mean": [0.5, 0.7], "sd": [0.1, 0.15],
                  "0.025quant": [0.3, 0.4], "0.975quant": [0.8, 0.95]}
    hyper_df = pd.DataFrame(hyper_rows, index=["sigma", "rho"])

    result = MagicMock()
    result.summary.random = {"area_id": random_df}
    result.summary.fixed = fixed_df
    result.summary.hyperpar = hyper_df
    result.dic = 234.5
    result.waic = 238.1
    result.marginals = {}

    return result


# ---------------------------------------------------------------------------
# _write_adj_file
# ---------------------------------------------------------------------------

class TestWriteAdjFile:
    def test_first_line_is_node_count(self, grid_3x3):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.adj")
            _write_adj_file(grid_3x3, path)
            with open(path) as f:
                lines = f.read().strip().split("\n")
        assert lines[0] == str(grid_3x3.n)

    def test_correct_number_of_node_lines(self, grid_3x3):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.adj")
            _write_adj_file(grid_3x3, path)
            with open(path) as f:
                lines = f.read().strip().split("\n")
        # First line is N, then one line per node
        assert len(lines) == grid_3x3.n + 1

    def test_nodes_are_one_indexed(self, grid_3x3):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.adj")
            _write_adj_file(grid_3x3, path)
            with open(path) as f:
                lines = f.read().strip().split("\n")
        # Second line should start with node id "1"
        first_node_line = lines[1].split()
        assert first_node_line[0] == "1"

    def test_all_neighbours_are_one_indexed(self, grid_3x3):
        """No neighbour index should be 0 (0-indexed would be a bug)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.adj")
            _write_adj_file(grid_3x3, path)
            with open(path) as f:
                lines = f.read().strip().split("\n")

        for line in lines[1:]:
            parts = line.split()
            node_id = int(parts[0])
            n_neigh = int(parts[1])
            neighbours = [int(x) for x in parts[2:]]
            assert len(neighbours) == n_neigh
            assert all(n >= 1 for n in neighbours), (
                f"Node {node_id}: neighbour indices must be >= 1 (1-indexed), got {neighbours}"
            )

    def test_symmetry_of_adjacency(self, grid_3x3):
        """If i -> j, then j -> i must also appear (symmetric graph)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.adj")
            _write_adj_file(grid_3x3, path)
            with open(path) as f:
                lines = f.read().strip().split("\n")

        adjacency_set: dict[int, set[int]] = {}
        for line in lines[1:]:
            parts = line.split()
            node_id = int(parts[0])
            neighbours = {int(x) for x in parts[2:]}
            adjacency_set[node_id] = neighbours

        for node, neighbours in adjacency_set.items():
            for neigh in neighbours:
                assert node in adjacency_set[neigh], (
                    f"Adjacency asymmetry: {node} -> {neigh} but not {neigh} -> {node}"
                )

    def test_corner_node_has_two_rook_neighbours(self, grid_3x3):
        """Corner node (1,1) in 3x3 rook grid has exactly 2 neighbours."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.adj")
            _write_adj_file(grid_3x3, path)
            with open(path) as f:
                lines = f.read().strip().split("\n")
        # Node 1 is the top-left corner in rook 3x3 → 2 neighbours
        corner_line = lines[1].split()
        assert corner_line[0] == "1"
        assert int(corner_line[1]) == 2

    def test_5x5_grid_total_edge_count(self, grid_5x5):
        """5×5 rook grid: 40 undirected edges → each node line lists directed edges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.adj")
            _write_adj_file(grid_5x5, path)
            with open(path) as f:
                lines = f.read().strip().split("\n")

        total_directed = sum(int(line.split()[1]) for line in lines[1:])
        # 40 undirected edges → 80 directed
        assert total_directed == 80


# ---------------------------------------------------------------------------
# _parse_summary_df
# ---------------------------------------------------------------------------

class TestParseSummaryDf:
    def test_returns_polars_dataframe(self):
        import pandas as pd
        raw = pd.DataFrame(
            {"mean": [0.1], "sd": [0.05], "0.025quant": [0.0], "0.975quant": [0.2]},
            index=["alpha"]
        )
        result = _parse_summary_df(raw, "parameter")
        assert isinstance(result, pl.DataFrame)

    def test_columns_present(self):
        import pandas as pd
        raw = pd.DataFrame(
            {"mean": [0.1, 0.5], "sd": [0.05, 0.1], "0.025quant": [0.0, 0.3], "0.975quant": [0.2, 0.7]},
            index=["alpha", "x1"]
        )
        result = _parse_summary_df(raw, "parameter")
        assert set(result.columns) == {"parameter", "mean", "sd", "q025", "q975"}

    def test_index_becomes_id_column(self):
        import pandas as pd
        raw = pd.DataFrame(
            {"mean": [0.1, 0.5], "sd": [0.05, 0.1], "0.025quant": [0.0, 0.3], "0.975quant": [0.2, 0.7]},
            index=["alpha", "x1"]
        )
        result = _parse_summary_df(raw, "parameter")
        assert result["parameter"].to_list() == ["alpha", "x1"]

    def test_none_input_returns_empty_dataframe(self):
        result = _parse_summary_df(None, "fixed_effect")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_missing_quantile_columns_fill_nan(self):
        import pandas as pd
        # DataFrame without quantile columns
        raw = pd.DataFrame({"mean": [0.3], "sd": [0.1]}, index=["sigma"])
        result = _parse_summary_df(raw, "parameter")
        assert result["q025"].is_nan().all()
        assert result["q975"].is_nan().all()


# ---------------------------------------------------------------------------
# BYM2InlaModel input validation (no pyINLA needed)
# ---------------------------------------------------------------------------

class TestBYM2InlaModelValidation:
    """
    Input validation tests — all run locally without pyINLA.
    We mock pyinla.inla to avoid the import.
    """

    def test_wrong_claims_length_raises(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        with pytest.raises(ValueError, match="claims"):
            with patch.dict("sys.modules", {"pyinla": MagicMock()}):
                # Patch pyinla but still hit validation before the pyinla call
                model.fit(claims=np.ones(10, dtype=int), exposure=exposure)

    def test_wrong_exposure_length_raises(self, grid_5x5, synthetic_data_5x5):
        claims, _ = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        with pytest.raises(ValueError, match="exposure"):
            with patch.dict("sys.modules", {"pyinla": MagicMock()}):
                model.fit(claims=claims, exposure=np.ones(10))

    def test_zero_exposure_raises(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        bad_exposure = exposure.copy()
        bad_exposure[0] = 0.0
        model = BYM2InlaModel(adjacency=grid_5x5)
        with pytest.raises(ValueError, match="exposure"):
            with patch.dict("sys.modules", {"pyinla": MagicMock()}):
                model.fit(claims=claims, exposure=bad_exposure)

    def test_negative_claims_raises(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        bad_claims = claims.copy()
        bad_claims[0] = -1
        model = BYM2InlaModel(adjacency=grid_5x5)
        with pytest.raises(ValueError, match="claim"):
            with patch.dict("sys.modules", {"pyinla": MagicMock()}):
                model.fit(claims=bad_claims, exposure=exposure)

    def test_wrong_covariates_length_raises(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        covariates = np.ones((10, 2))  # wrong N
        model = BYM2InlaModel(adjacency=grid_5x5)
        with pytest.raises(ValueError, match="covariates"):
            with patch.dict("sys.modules", {"pyinla": MagicMock()}):
                model.fit(claims=claims, exposure=exposure, covariates=covariates)

    def test_covariate_names_length_mismatch_raises(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        N = grid_5x5.n
        covariates = np.ones((N, 2))
        model = BYM2InlaModel(adjacency=grid_5x5, covariate_names=["only_one"])
        # covariate_names has 1 entry but covariates has 2 columns
        # This is caught in fit() after the pyINLA import check
        pyinla_mock = MagicMock()
        pyinla_mock.Formula = MagicMock()
        pyinla_mock.inla = MagicMock()
        with pytest.raises(ValueError, match="covariate_names"):
            with patch.dict("sys.modules", {"pyinla": pyinla_mock}):
                model.fit(claims=claims, exposure=exposure, covariates=covariates)

    def test_missing_pyinla_raises_import_error(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        with patch.dict("sys.modules", {"pyinla": None}):
            with pytest.raises(ImportError, match="pyinla"):
                model.fit(claims=claims, exposure=exposure)


# ---------------------------------------------------------------------------
# BYM2InlaModel.fit() with mocked pyINLA
# ---------------------------------------------------------------------------

class TestBYM2InlaModelFitMocked:
    """
    Test BYM2InlaModel.fit() end-to-end, mocking the pyinla.inla() call.

    This tests our DataFrame assembly, .adj writing, and result parsing.
    """

    def _run_fit(self, model, claims, exposure, covariates=None):
        """Run fit() with pyINLA mocked out."""
        import pandas as pd

        N = model.adjacency.n
        mock_result = _make_mock_inla_result(N, n_fixed=(0 if covariates is None else covariates.shape[1]))

        pyinla_mock = MagicMock()
        pyinla_mock.Formula = MagicMock(return_value=MagicMock())
        pyinla_mock.inla = MagicMock(return_value=mock_result)

        with patch.dict("sys.modules", {"pyinla": pyinla_mock, "pandas": pd}):
            result = model.fit(claims=claims, exposure=exposure, covariates=covariates)

        return result, pyinla_mock

    def test_returns_bym2_inla_result(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result, _ = self._run_fit(model, claims, exposure)
        assert isinstance(result, BYM2InlaResult)

    def test_b_mean_shape(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result, _ = self._run_fit(model, claims, exposure)
        assert result.b_mean.shape == (grid_5x5.n,)

    def test_b_sd_shape(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result, _ = self._run_fit(model, claims, exposure)
        assert result.b_sd.shape == (grid_5x5.n,)

    def test_n_areas_correct(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result, _ = self._run_fit(model, claims, exposure)
        assert result.n_areas == grid_5x5.n

    def test_areas_match_adjacency(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result, _ = self._run_fit(model, claims, exposure)
        assert result.areas == list(grid_5x5.areas)

    def test_pyinla_inla_called_once(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        _, pyinla_mock = self._run_fit(model, claims, exposure)
        pyinla_mock.inla.assert_called_once()

    def test_adj_file_written_to_tempdir(self, grid_5x5, synthetic_data_5x5):
        """
        The .adj path passed to pyINLA Formula should be a real file path
        (even though pyINLA is mocked, we can verify the path argument).
        """
        import pandas as pd
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)

        N = grid_5x5.n
        mock_result = _make_mock_inla_result(N)
        captured_formula_kwargs: dict = {}

        def capture_formula(**kwargs):
            captured_formula_kwargs.update(kwargs)
            return MagicMock()

        pyinla_mock = MagicMock()
        pyinla_mock.Formula = MagicMock(side_effect=capture_formula)
        pyinla_mock.inla = MagicMock(return_value=mock_result)

        with patch.dict("sys.modules", {"pyinla": pyinla_mock, "pandas": pd}):
            model.fit(claims=claims, exposure=exposure)

        # Extract graph path from random_effects spec
        random_effects = captured_formula_kwargs.get("random_effects", [])
        assert len(random_effects) == 1
        graph_path = random_effects[0]["graph"]
        # The .adj file was in a tempdir that is cleaned up after fit() returns,
        # so we can only verify it was a string path ending in .adj
        assert graph_path.endswith(".adj")

    def test_dic_and_waic_populated(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result, _ = self._run_fit(model, claims, exposure)
        assert result.dic == pytest.approx(234.5)
        assert result.waic == pytest.approx(238.1)

    def test_fit_with_covariates(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        N = grid_5x5.n
        rng = np.random.default_rng(77)
        covariates = rng.standard_normal((N, 2))
        model = BYM2InlaModel(adjacency=grid_5x5)
        result, _ = self._run_fit(model, claims, exposure, covariates=covariates)
        assert isinstance(result, BYM2InlaResult)

    def test_fit_1d_covariate_broadcast(self, grid_5x5, synthetic_data_5x5):
        """1D covariate array should be reshaped to (N, 1) without error."""
        claims, exposure = synthetic_data_5x5
        N = grid_5x5.n
        rng = np.random.default_rng(88)
        covariates_1d = rng.standard_normal(N)
        model = BYM2InlaModel(adjacency=grid_5x5)
        result, _ = self._run_fit(model, claims, exposure, covariates=covariates_1d)
        assert isinstance(result, BYM2InlaResult)


# ---------------------------------------------------------------------------
# BYM2InlaResult.territory_relativities()
# ---------------------------------------------------------------------------

class TestBYM2InlaResultRelativities:

    @pytest.fixture
    def fitted_result(self, grid_5x5) -> BYM2InlaResult:
        """Build a BYM2InlaResult directly with known b values."""
        N = grid_5x5.n
        rng = np.random.default_rng(123)
        b_mean = rng.standard_normal(N) * 0.3
        b_sd = np.abs(rng.standard_normal(N)) * 0.05 + 0.02

        fixed_summary = pl.DataFrame({
            "fixed_effect": ["alpha"],
            "mean": [0.0],
            "sd": [0.1],
            "q025": [-0.2],
            "q975": [0.2],
        })
        hyperpar_summary = pl.DataFrame({
            "parameter": ["sigma", "rho"],
            "mean": [0.5, 0.7],
            "sd": [0.1, 0.15],
            "q025": [0.3, 0.4],
            "q975": [0.8, 0.95],
        })

        return BYM2InlaResult(
            b_mean=b_mean,
            b_sd=b_sd,
            areas=list(grid_5x5.areas),
            adjacency=grid_5x5,
            n_areas=N,
            fixed_summary=fixed_summary,
            hyperpar_summary=hyperpar_summary,
            dic=200.0,
            waic=205.0,
        )

    def test_returns_polars_dataframe(self, fitted_result):
        rels = fitted_result.territory_relativities()
        assert isinstance(rels, pl.DataFrame)

    def test_correct_number_of_rows(self, fitted_result, grid_5x5):
        rels = fitted_result.territory_relativities()
        assert len(rels) == grid_5x5.n

    def test_expected_columns(self, fitted_result):
        rels = fitted_result.territory_relativities()
        expected = {"area", "b_mean", "b_sd", "relativity", "lower", "upper", "ln_offset"}
        assert set(rels.columns) == expected

    def test_relativities_positive(self, fitted_result):
        rels = fitted_result.territory_relativities()
        assert (rels["relativity"] > 0).all()

    def test_lower_le_relativity_le_upper(self, fitted_result):
        rels = fitted_result.territory_relativities()
        assert (rels["lower"] <= rels["relativity"]).all()
        assert (rels["relativity"] <= rels["upper"]).all()

    def test_grand_mean_normalisation(self, fitted_result):
        """Without base_area, relativities should have geometric mean ~1.0."""
        rels = fitted_result.territory_relativities()
        # Geometric mean = exp(mean(log(rel)))
        log_rels = np.log(rels["relativity"].to_numpy())
        # ln_offset is b_mean - mean(b_mean), so sum should be ~0
        ln_offsets = rels["ln_offset"].to_numpy()
        assert abs(ln_offsets.mean()) < 1e-10

    def test_base_area_has_relativity_one(self, fitted_result, grid_5x5):
        base = grid_5x5.areas[0]
        rels = fitted_result.territory_relativities(base_area=base)
        base_row = rels.filter(pl.col("area") == base)
        assert base_row["relativity"][0] == pytest.approx(1.0, abs=1e-10)

    def test_base_area_not_found_raises(self, fitted_result):
        with pytest.raises(ValueError, match="base_area"):
            fitted_result.territory_relativities(base_area="nonexistent_area")

    def test_wider_interval_with_higher_credibility(self, fitted_result):
        rels_95 = fitted_result.territory_relativities(credibility_interval=0.95)
        rels_80 = fitted_result.territory_relativities(credibility_interval=0.80)
        width_95 = (rels_95["upper"] - rels_95["lower"]).mean()
        width_80 = (rels_80["upper"] - rels_80["lower"]).mean()
        assert width_95 > width_80

    def test_ln_offset_is_b_mean_minus_reference(self, fitted_result, grid_5x5):
        """ln_offset should equal b_mean - mean(b_mean) for grand mean normalisation."""
        rels = fitted_result.territory_relativities()
        b_mean = fitted_result.b_mean
        expected_ln_offset = b_mean - b_mean.mean()
        actual_ln_offset = rels["ln_offset"].to_numpy()
        np.testing.assert_allclose(actual_ln_offset, expected_ln_offset, atol=1e-10)

    def test_areas_match_adjacency_order(self, fitted_result, grid_5x5):
        rels = fitted_result.territory_relativities()
        assert rels["area"].to_list() == list(grid_5x5.areas)


# ---------------------------------------------------------------------------
# BYM2InlaResult.diagnostics()
# ---------------------------------------------------------------------------

class TestBYM2InlaResultDiagnostics:

    @pytest.fixture
    def minimal_result(self, grid_3x3) -> BYM2InlaResult:
        N = grid_3x3.n
        return BYM2InlaResult(
            b_mean=np.zeros(N),
            b_sd=np.ones(N) * 0.1,
            areas=list(grid_3x3.areas),
            adjacency=grid_3x3,
            n_areas=N,
            fixed_summary=pl.DataFrame({"fixed_effect": [], "mean": [], "sd": [], "q025": [], "q975": []}),
            hyperpar_summary=pl.DataFrame({"parameter": ["sigma", "rho"],
                                           "mean": [0.4, 0.6], "sd": [0.1, 0.1],
                                           "q025": [0.2, 0.4], "q975": [0.7, 0.9]}),
            dic=150.0,
            waic=155.0,
        )

    def test_returns_inla_diagnostics(self, minimal_result):
        diag = minimal_result.diagnostics()
        assert isinstance(diag, InlaDiagnostics)

    def test_dic_correct(self, minimal_result):
        diag = minimal_result.diagnostics()
        assert diag.dic == pytest.approx(150.0)

    def test_waic_correct(self, minimal_result):
        diag = minimal_result.diagnostics()
        assert diag.waic == pytest.approx(155.0)

    def test_hyperpar_summary_is_dataframe(self, minimal_result):
        diag = minimal_result.diagnostics()
        assert isinstance(diag.hyperpar_summary, pl.DataFrame)


# ---------------------------------------------------------------------------
# Full integration test (skipped without pyINLA)
# ---------------------------------------------------------------------------

try:
    import pyinla as _pyinla_check  # noqa: F401
    PYINLA_AVAILABLE = True
except ImportError:
    PYINLA_AVAILABLE = False

requires_pyinla = pytest.mark.skipif(
    not PYINLA_AVAILABLE,
    reason=(
        "pyINLA not installed. "
        "Install with: uv add 'insurance-spatial[inla]' (Linux x86_64 / macOS only). "
        "Run on Databricks for full integration tests."
    )
)


@requires_pyinla
class TestBYM2InlaIntegration:
    """
    Full integration tests that call pyinla.inla() for real.

    Designed to run on Databricks (x86_64 Linux) where pyINLA is available.
    On ARM64 Linux (Raspberry Pi), these tests are skipped.

    We use a 5×5 grid (N=25) with synthetic Poisson data.  The INLA
    computation should complete in well under 1 second.
    """

    def test_fit_returns_bym2_inla_result(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result = model.fit(claims=claims, exposure=exposure)
        assert isinstance(result, BYM2InlaResult)

    def test_b_mean_length(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result = model.fit(claims=claims, exposure=exposure)
        assert len(result.b_mean) == grid_5x5.n

    def test_territory_relativities_schema(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result = model.fit(claims=claims, exposure=exposure)
        rels = result.territory_relativities()
        expected_cols = {"area", "b_mean", "b_sd", "relativity", "lower", "upper", "ln_offset"}
        assert set(rels.columns) == expected_cols
        assert len(rels) == grid_5x5.n

    def test_relativities_positive(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result = model.fit(claims=claims, exposure=exposure)
        rels = result.territory_relativities()
        assert (rels["relativity"] > 0).all()

    def test_hyperpar_summary_has_sigma_and_rho(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        result = model.fit(claims=claims, exposure=exposure)
        param_names = result.hyperpar_summary["parameter"].to_list()
        # We expect sigma and rho (possibly under different R-INLA naming)
        assert len(param_names) >= 2

    def test_fit_with_covariates_runs(self, grid_5x5, synthetic_data_5x5):
        claims, exposure = synthetic_data_5x5
        N = grid_5x5.n
        rng = np.random.default_rng(55)
        covariates = rng.standard_normal((N, 1))
        model = BYM2InlaModel(adjacency=grid_5x5, covariate_names=["urban_score"])
        result = model.fit(claims=claims, exposure=exposure, covariates=covariates)
        assert isinstance(result, BYM2InlaResult)

    def test_inla_faster_than_5_seconds_on_25_areas(self, grid_5x5, synthetic_data_5x5):
        """pyINLA on N=25 should complete in well under 5 seconds."""
        import time
        claims, exposure = synthetic_data_5x5
        model = BYM2InlaModel(adjacency=grid_5x5)
        t0 = time.perf_counter()
        model.fit(claims=claims, exposure=exposure)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"pyINLA took {elapsed:.1f}s on N=25, expected < 5s"
