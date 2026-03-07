"""
Integration tests for BYM2 model fitting.

These tests require PyMC and run an actual MCMC chain.  They are designed to
run on Databricks (where PyMC is available) and are marked accordingly.
On the local Raspberry Pi, skip them to avoid crashing the machine.

Run on Databricks:
    databricks jobs run-now --job-id <job-id>
or via the demo notebook at notebooks/bym2_demo.py.
"""

import numpy as np
import polars as pl
import pytest

from insurance_spatial.adjacency import build_grid_adjacency
from insurance_spatial.models import BYM2Model
from insurance_spatial.diagnostics import convergence_summary


# Skip by default unless explicitly running in Databricks / CI environment
SKIP_HEAVY = True
try:
    import pymc  # noqa: F401
    SKIP_HEAVY = False
except ImportError:
    SKIP_HEAVY = True

requires_pymc = pytest.mark.skipif(
    SKIP_HEAVY,
    reason="PyMC not installed. Run on Databricks: uv add pymc"
)


@requires_pymc
class TestBYM2ModelFit:
    """
    Full fit tests on the 5×5 synthetic grid.

    We use very short chains (draws=100, tune=200, chains=2) to keep runtime
    manageable.  Convergence checks are therefore lenient.
    """

    def test_fit_returns_bym2_result(self, grid_5x5, synthetic_claims_5x5):
        claims, exposure, _ = synthetic_claims_5x5
        model = BYM2Model(
            adjacency=grid_5x5,
            draws=100,
            chains=2,
            tune=200,
        )
        result = model.fit(claims=claims, exposure=exposure, random_seed=42)
        from insurance_spatial.models import BYM2Result
        assert isinstance(result, BYM2Result)

    def test_trace_has_expected_variables(self, grid_5x5, synthetic_claims_5x5):
        claims, exposure, _ = synthetic_claims_5x5
        model = BYM2Model(adjacency=grid_5x5, draws=100, chains=2, tune=200)
        result = model.fit(claims=claims, exposure=exposure, random_seed=42)
        posterior_vars = list(result.trace.posterior.data_vars)
        for var in ["alpha", "sigma", "rho", "phi", "theta", "b"]:
            assert var in posterior_vars, f"Missing variable: {var}"

    def test_rho_in_unit_interval(self, grid_5x5, synthetic_claims_5x5):
        claims, exposure, _ = synthetic_claims_5x5
        model = BYM2Model(adjacency=grid_5x5, draws=100, chains=2, tune=200)
        result = model.fit(claims=claims, exposure=exposure, random_seed=42)
        rho_samples = result.trace.posterior["rho"].values.ravel()
        assert np.all(rho_samples >= 0.0)
        assert np.all(rho_samples <= 1.0)

    def test_b_shape(self, grid_5x5, synthetic_claims_5x5):
        claims, exposure, _ = synthetic_claims_5x5
        N = grid_5x5.n
        model = BYM2Model(adjacency=grid_5x5, draws=100, chains=2, tune=200)
        result = model.fit(claims=claims, exposure=exposure, random_seed=42)
        b_shape = result.trace.posterior["b"].shape
        # (chains, draws, N)
        assert b_shape == (2, 100, N)

    def test_territory_relativities_output(self, grid_5x5, synthetic_claims_5x5):
        claims, exposure, _ = synthetic_claims_5x5
        model = BYM2Model(adjacency=grid_5x5, draws=100, chains=2, tune=200)
        result = model.fit(claims=claims, exposure=exposure, random_seed=42)
        rels = result.territory_relativities()
        assert isinstance(rels, pl.DataFrame)
        assert len(rels) == grid_5x5.n
        assert (rels["relativity"] > 0).all()
        assert (rels["lower"] <= rels["upper"]).all()

    def test_fit_with_covariates(self, grid_5x5, synthetic_claims_5x5):
        claims, exposure, _ = synthetic_claims_5x5
        N = grid_5x5.n
        rng = np.random.default_rng(99)
        covariates = rng.standard_normal((N, 2))
        model = BYM2Model(adjacency=grid_5x5, draws=100, chains=2, tune=200)
        result = model.fit(
            claims=claims,
            exposure=exposure,
            covariates=covariates,
            random_seed=42,
        )
        posterior_vars = list(result.trace.posterior.data_vars)
        assert "beta" in posterior_vars
        assert result.trace.posterior["beta"].shape[-1] == 2

    def test_wrong_claims_length_raises(self, grid_5x5):
        model = BYM2Model(adjacency=grid_5x5, draws=50, chains=1)
        with pytest.raises(ValueError, match="claims"):
            model.fit(
                claims=np.ones(10, dtype=int),  # wrong length
                exposure=np.ones(25),
            )

    def test_zero_exposure_raises(self, grid_5x5):
        model = BYM2Model(adjacency=grid_5x5, draws=50, chains=1)
        exposure = np.ones(25)
        exposure[0] = 0.0
        with pytest.raises(ValueError, match="exposure"):
            model.fit(claims=np.ones(25, dtype=int), exposure=exposure)

    def test_negative_claims_raises(self, grid_5x5):
        model = BYM2Model(adjacency=grid_5x5, draws=50, chains=1)
        claims = np.ones(25, dtype=int)
        claims[0] = -1
        with pytest.raises(ValueError, match="claim"):
            model.fit(claims=claims, exposure=np.ones(25))

    def test_convergence_summary_runs(self, grid_5x5, synthetic_claims_5x5):
        claims, exposure, _ = synthetic_claims_5x5
        model = BYM2Model(adjacency=grid_5x5, draws=200, chains=2, tune=300)
        result = model.fit(claims=claims, exposure=exposure, random_seed=42)
        from insurance_spatial.diagnostics import ConvergenceSummary
        summary = convergence_summary(result)
        assert isinstance(summary, ConvergenceSummary)
        assert isinstance(summary.max_rhat, float)
        assert isinstance(summary.n_divergences, int)
