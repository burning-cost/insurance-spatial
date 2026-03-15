"""
BYM2 spatial territory model.

The model
---------
We fit a Poisson frequency model with a BYM2 spatial random effect:

    y_i ~ Poisson(mu_i)
    log(mu_i) = log(E_i) + alpha + X_i @ beta + b_i

    b_i = sigma * (sqrt(rho) * phi*_i + sqrt(1 - rho) * theta_i)

    where phi*_i = phi_i / sqrt(s)  (pre-scaled to unit marginal variance)

    phi ~ ICAR(W)          # structured spatial component
    theta ~ Normal(0, 1)   # unstructured (IID) component
    sigma ~ HalfNormal(1)
    rho ~ Beta(0.5, 0.5)
    alpha ~ Normal(0, 1)
    beta ~ Normal(0, 1)

Here:
  - E_i is exposure (policy-years) for area i, used as an offset
  - s is the BYM2 scaling factor (geometric mean of ICAR marginal variances)
  - phi* = phi / sqrt(s) has unit marginal variance after scaling
  - rho is the proportion of variance attributable to spatial structure

The rho parameter is directly interpretable: rho near 1 means the residual
geographic variation is spatially smooth; rho near 0 means it is pure noise.

Parameterisation note (Riebler et al. 2016)
--------------------------------------------
The critical step is to pre-scale phi by 1/sqrt(s) BEFORE applying the rho
mixture weight.  This ensures that Var(phi*) = 1 = Var(theta), so rho is the
true proportion of marginal variance from the spatial component.  Writing the
formula as sqrt(rho/s) * phi (without pre-scaling) is algebraically equivalent
for the point estimate of b, but rho loses its variance-proportion interpretation
because the effective spatial weight depends on both rho and s together.

Optional nutpie sampler
-----------------------
nutpie is a Rust-based NUTS implementation that is typically 2-5x faster than
PyMC's default NUTS for models of this type.  We use it when available and fall
back to PyMC's default sampler with a warning.

Two-stage vs. integrated pipeline
----------------------------------
You can use this model in two ways:

1. Integrated: pass raw claims and exposure.  The model captures all geographic
   variation in the residuals after the covariate effects (beta).

2. Two-stage (recommended for production): fit a non-spatial GLM or GBM first,
   compute sector-level O/E ratios, then pass those O/E ratios as the input here.
   This keeps the spatial model decoupled and easier to explain at actuarial review.
   In this case, pass observed=oe_claims, exposure=oe_exposure where oe_claims is
   sector-level observed claim count and oe_exposure is expected claims from the
   base model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import polars as pl
from scipy import sparse

from insurance_spatial.adjacency import AdjacencyMatrix


@dataclass
class BYM2Result:
    """
    Container for the output of a fitted BYM2Model.

    Attributes
    ----------
    trace :
        ArviZ InferenceData object containing the full MCMC trace.
    areas :
        Ordered list of area identifiers corresponding to model indices.
    adjacency :
        The adjacency structure used in fitting.
    n_areas :
        Number of areas (N).
    """

    trace: object  # arviz.InferenceData
    areas: list[str]
    adjacency: AdjacencyMatrix
    n_areas: int

    def territory_relativities(
        self,
        base_area: Optional[str] = None,
        credibility_interval: float = 0.95,
    ) -> pl.DataFrame:
        """
        Extract territory relativities from the fitted model.

        Returns a Polars DataFrame with columns:
          - ``area``: area identifier
          - ``relativity``: posterior mean of exp(b_i), normalised to grand mean 1.0
            (or relative to ``base_area`` if specified)
          - ``lower``, ``upper``: credibility interval bounds on the relativity
          - ``b_mean``: posterior mean of the log-scale spatial effect b_i
          - ``b_sd``: posterior standard deviation of b_i

        Parameters
        ----------
        base_area :
            If provided, normalise all relativities so that this area has
            relativity 1.0. Otherwise normalise to the geometric mean
            (sum-to-zero on log scale).
        credibility_interval :
            Width of the symmetric posterior interval.  Default 0.95 (95% CI).
        """
        from insurance_spatial.relativities import extract_relativities
        return extract_relativities(
            self,
            base_area=base_area,
            credibility_interval=credibility_interval,
        )

    def diagnostics(self) -> "SpatialDiagnostics":
        """
        Return convergence and spatial autocorrelation diagnostics.

        See :mod:`insurance_spatial.diagnostics` for full detail.
        """
        from insurance_spatial.diagnostics import compute_diagnostics
        return compute_diagnostics(self)


class BYM2Model:
    """
    BYM2 spatial Poisson model for territory ratemaking.

    Parameters
    ----------
    adjacency :
        AdjacencyMatrix built from your territory boundary data.
    draws :
        MCMC draws per chain (post-warmup).
    chains :
        Number of MCMC chains.
    target_accept :
        Target acceptance rate for NUTS.  0.9 is a reasonable default for
        spatial models; increase to 0.95 if you see divergences.
    tune :
        Number of tuning (warmup) steps per chain.

    Example
    -------
    >>> from insurance_spatial import build_grid_adjacency, BYM2Model
    >>> adj = build_grid_adjacency(5, 5)
    >>> model = BYM2Model(adjacency=adj, draws=500, chains=2)
    >>> result = model.fit(claims=counts, exposure=exposure)
    >>> rels = result.territory_relativities()
    """

    def __init__(
        self,
        adjacency: AdjacencyMatrix,
        draws: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        tune: int = 1000,
    ) -> None:
        self.adjacency = adjacency
        self.draws = draws
        self.chains = chains
        self.target_accept = target_accept
        self.tune = tune

    def fit(
        self,
        claims: np.ndarray,
        exposure: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ) -> BYM2Result:
        """
        Fit the BYM2 model via MCMC.

        Parameters
        ----------
        claims :
            Array of shape (N,) containing observed claim counts per area.
            Must be non-negative integers.
        exposure :
            Array of shape (N,) containing exposure (e.g. policy-years) per area.
            Must be strictly positive.
        covariates :
            Optional float array of shape (N, P) containing area-level covariates
            (e.g. IMD score, crime rate).  These enter as fixed effects (beta).
            Scale them before passing - the prior on beta is Normal(0, 1).
        random_seed :
            Integer seed for reproducibility.

        Returns
        -------
        BYM2Result
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError as exc:
            raise ImportError(
                "Fitting requires pymc. Install with: uv add pymc"
            ) from exc

        claims = np.asarray(claims, dtype=np.int64)
        exposure = np.asarray(exposure, dtype=np.float64)

        if claims.shape[0] != self.adjacency.n:
            raise ValueError(
                f"claims has {claims.shape[0]} entries but adjacency has {self.adjacency.n} areas"
            )
        if exposure.shape[0] != self.adjacency.n:
            raise ValueError(
                f"exposure has {exposure.shape[0]} entries but adjacency has {self.adjacency.n} areas"
            )
        if np.any(exposure <= 0):
            raise ValueError("All exposure values must be strictly positive")
        if np.any(claims < 0):
            raise ValueError("claim counts must be non-negative")

        N = self.adjacency.n
        W_dense = self.adjacency.to_dense()
        scaling_factor = self.adjacency.scaling_factor
        log_E = np.log(exposure)

        if covariates is not None:
            covariates = np.asarray(covariates, dtype=np.float64)
            if covariates.ndim == 1:
                covariates = covariates[:, None]
            if covariates.shape[0] != N:
                raise ValueError(
                    f"covariates has {covariates.shape[0]} rows but {N} areas"
                )

        nuts_sampler, sampler_kwargs = _resolve_sampler()

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

            if covariates is not None:
                P = covariates.shape[1]
                beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=P)
                linear_pred = alpha + pt.dot(covariates, beta)
            else:
                linear_pred = alpha

            # BYM2 hyperpriors
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            rho = pm.Beta("rho", alpha=0.5, beta=0.5)

            # Spatial components
            phi = pm.ICAR("phi", W=W_dense, shape=N)
            theta = pm.Normal("theta", mu=0.0, sigma=1.0, shape=N)

            # BYM2 combined spatial effect (Riebler et al. 2016, eq. 6)
            # Pre-scale phi so it has unit marginal variance, THEN apply rho
            # as a mixture weight.  This preserves rho's interpretation as the
            # proportion of marginal variance attributable to spatial structure.
            # Using sqrt(rho / s) * phi is algebraically equivalent for b but
            # breaks the rho interpretation when scaling_factor != 1.
            phi_scaled = phi / pt.sqrt(float(scaling_factor))
            b = pm.Deterministic(
                "b",
                sigma * (
                    pt.sqrt(rho) * phi_scaled
                    + pt.sqrt(1.0 - rho) * theta
                ),
            )

            # Poisson likelihood
            log_mu = log_E + linear_pred + b
            mu = pm.Deterministic("mu", pt.exp(log_mu))
            pm.Poisson("y", mu=mu, observed=claims)

        with model:
            trace = pm.sample(
                draws=self.draws,
                chains=self.chains,
                tune=self.tune,
                target_accept=self.target_accept,
                nuts_sampler=nuts_sampler,
                random_seed=random_seed,
                **sampler_kwargs,
            )

        return BYM2Result(
            trace=trace,
            areas=list(self.adjacency.areas),
            adjacency=self.adjacency,
            n_areas=N,
        )


def _resolve_sampler() -> tuple[str, dict]:
    """
    Detect whether nutpie is available and return sampler name + extra kwargs.
    Falls back to PyMC default NUTS with a warning.
    """
    try:
        import nutpie  # noqa: F401
        return "nutpie", {}
    except ImportError:
        warnings.warn(
            "nutpie is not installed. Using PyMC's default NUTS sampler. "
            "Install nutpie for 2-5x faster sampling: uv add nutpie",
            UserWarning,
            stacklevel=3,
        )
        return "pymc", {}
