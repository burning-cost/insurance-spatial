"""
Spatial and MCMC convergence diagnostics.

Moran's I
---------
Moran's I is the standard test for spatial autocorrelation.  Run it on your
territory residuals (observed / expected) before fitting BYM2 to confirm that
spatial smoothing is warranted.  Run it again on posterior predictive residuals
to confirm that the model has absorbed the spatial structure.

A positive, significant Moran's I means nearby areas have similar residuals —
spatial autocorrelation is present and BYM2 is appropriate.
A non-significant Moran's I post-fit means the spatial model has adequately
captured geographic variation.

Convergence diagnostics
------------------------
We use ArviZ's R-hat (Gelman-Rubin) and ESS (effective sample size) statistics.
Thresholds:
  - R-hat < 1.01 for all parameters (strict)
  - ESS > 400 per parameter per chain (bulk and tail)

These are the thresholds recommended in Vehtari et al. (2021) "Rank-normalization,
folding, and localization: An improved R-hat for assessing convergence of MCMC".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from insurance_spatial.models import BYM2Result
    from insurance_spatial.adjacency import AdjacencyMatrix


@dataclass
class MoranI:
    """
    Result of a Moran's I spatial autocorrelation test.

    Attributes
    ----------
    statistic :
        Moran's I statistic.  Ranges from -1 (perfect negative) to +1 (perfect positive).
    expected :
        Expected value under the null hypothesis of no spatial autocorrelation.
        For N areas with row-standardised weights: E[I] = -1/(N-1).
    p_value :
        Two-tailed pseudo p-value from permutation test (more reliable than
        analytical approximation).  Significant for both positive and negative
        autocorrelation.
    z_score :
        Z-score from permutation distribution.
    n_permutations :
        Number of permutations used.
    significant :
        True if p_value < 0.05.
    interpretation :
        Plain-English summary.
    """

    statistic: float
    expected: float
    p_value: float
    z_score: float
    n_permutations: int
    significant: bool
    interpretation: str


@dataclass
class ConvergenceSummary:
    """
    MCMC convergence diagnostics.

    Attributes
    ----------
    max_rhat :
        Maximum R-hat across all scalar parameters.  Should be < 1.01.
    min_ess_bulk :
        Minimum bulk ESS across all scalar parameters.  Should be > 400.
    min_ess_tail :
        Minimum tail ESS.  Should be > 400.
    converged :
        True if max_rhat < 1.01 and min_ess_bulk > 400 and min_ess_tail > 400.
    rhat_by_param :
        Polars DataFrame with R-hat per named parameter group.
    ess_by_param :
        Polars DataFrame with bulk ESS per named parameter group.
    n_divergences :
        Total number of divergent transitions across all chains.
    """

    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    converged: bool
    rhat_by_param: pl.DataFrame
    ess_by_param: pl.DataFrame
    n_divergences: int


@dataclass
class SpatialDiagnostics:
    """
    Combined spatial and convergence diagnostics from a fitted BYM2 model.
    """

    convergence: ConvergenceSummary
    rho_summary: pl.DataFrame  # posterior of rho (spatial proportion)
    sigma_summary: pl.DataFrame  # posterior of sigma (total SD)
    moran_post: Optional[MoranI]  # Moran's I on posterior residuals (if computed)


def moran_i(
    values: np.ndarray,
    adjacency: "AdjacencyMatrix",
    n_permutations: int = 999,
) -> MoranI:
    """
    Compute Moran's I for a set of areal values given an adjacency structure.

    This is the pure-Python/numpy implementation that does not require esda.
    It computes the permutation-based p-value directly.

    The p-value is two-tailed: it reflects the probability of observing a
    statistic as extreme as the observed one in either direction under the
    null hypothesis of spatial randomness.  This correctly flags both
    significant positive autocorrelation (clustered) and significant negative
    autocorrelation (dispersed / checkerboard).

    Parameters
    ----------
    values :
        Array of shape (N,) - e.g. log(O/E) residuals per area.
    adjacency :
        AdjacencyMatrix with N areas matching the length of values.
    n_permutations :
        Number of random permutations for the null distribution.

    Returns
    -------
    MoranI
    """
    values = np.asarray(values, dtype=np.float64)
    N = len(values)
    if len(adjacency.areas) != N:
        raise ValueError(
            f"values has {N} entries but adjacency has {len(adjacency.areas)} areas"
        )

    # Row-standardise W
    W = adjacency.W.toarray().astype(np.float64)
    row_sums = W.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated nodes (should not occur after fix_islands)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    W_std = W / row_sums

    def _moran_stat(x: np.ndarray) -> float:
        z = x - x.mean()
        numerator = float(N * z @ W_std @ z)
        denominator = float(z @ z)
        S0 = float(W_std.sum())
        if denominator == 0 or S0 == 0:
            return 0.0
        return numerator / (S0 * denominator)

    observed_I = _moran_stat(values)

    # Permutation test
    rng = np.random.default_rng(42)
    perm_stats = np.array([_moran_stat(rng.permutation(values)) for _ in range(n_permutations)])

    expected_I = -1.0 / (N - 1)
    perm_mean = perm_stats.mean()
    perm_std = perm_stats.std()
    z_score = (observed_I - perm_mean) / perm_std if perm_std > 0 else 0.0

    # P0 fix: two-tailed p-value so that significant negative autocorrelation
    # (observed_I much smaller than permutation distribution) is correctly flagged.
    # A one-tailed test using perm_stats >= observed_I would give p ~ 1.0 for
    # strongly negative I, making the result look completely non-significant.
    p_value = float(np.mean(np.abs(perm_stats - perm_mean) >= abs(observed_I - perm_mean)))

    significant = p_value < 0.05
    if significant and observed_I > 0:
        interpretation = (
            f"Significant positive spatial autocorrelation (I={observed_I:.3f}, "
            f"p={p_value:.3f}). Nearby areas have similar values. "
            "Spatial smoothing is warranted."
        )
    elif significant and observed_I < 0:
        interpretation = (
            f"Significant negative spatial autocorrelation (I={observed_I:.3f}, "
            f"p={p_value:.3f}). Nearby areas have dissimilar values."
        )
    else:
        interpretation = (
            f"No significant spatial autocorrelation (I={observed_I:.3f}, "
            f"p={p_value:.3f}). Spatial smoothing may not be necessary."
        )

    return MoranI(
        statistic=observed_I,
        expected=expected_I,
        p_value=p_value,
        z_score=z_score,
        n_permutations=n_permutations,
        significant=significant,
        interpretation=interpretation,
    )


def convergence_summary(result: "BYM2Result") -> ConvergenceSummary:
    """
    Compute MCMC convergence diagnostics from a fitted BYM2Result.

    Parameters
    ----------
    result :
        Fitted BYM2Result.

    Returns
    -------
    ConvergenceSummary
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("convergence_summary requires arviz. uv add arviz") from exc

    trace = result.trace

    # Compute R-hat and ESS for all parameters
    rhat_data = az.rhat(trace)
    ess_data = az.ess(trace)

    param_names = []
    rhat_vals = []
    ess_bulk_vals = []

    # Scalar parameters of primary interest
    scalar_params = ["alpha", "sigma", "rho"]
    # Vector parameters - summarise by max R-hat / min ESS
    vector_params = ["phi", "theta", "b"]

    for param in scalar_params:
        if param in rhat_data:
            param_names.append(param)
            rhat_vals.append(float(rhat_data[param].values))
            ess_bulk_vals.append(float(ess_data[param].values) if param in ess_data else float("nan"))

    for param in vector_params:
        if param in rhat_data:
            rhat_arr = rhat_data[param].values.ravel()
            ess_arr = ess_data[param].values.ravel() if param in ess_data else np.array([float("nan")])
            param_names.append(f"{param}[max_rhat]")
            rhat_vals.append(float(np.nanmax(rhat_arr)))
            ess_bulk_vals.append(float(np.nanmin(ess_arr)))

    rhat_df = pl.DataFrame({
        "parameter": param_names,
        "rhat": rhat_vals,
        "ess_bulk": ess_bulk_vals,
    })

    max_rhat = float(np.nanmax(rhat_vals)) if rhat_vals else float("nan")
    min_ess_bulk = float(np.nanmin(ess_bulk_vals)) if ess_bulk_vals else float("nan")

    # Tail ESS (from "tail" kind)
    ess_tail_data = az.ess(trace, method="tail")
    tail_vals = []
    for param in scalar_params + vector_params:
        if param in ess_tail_data:
            arr = ess_tail_data[param].values.ravel()
            tail_vals.append(float(np.nanmin(arr)))
    min_ess_tail = float(np.nanmin(tail_vals)) if tail_vals else float("nan")

    # Divergences
    n_divergences = 0
    if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
        n_divergences = int(trace.sample_stats["diverging"].values.sum())

    # P1 fix: include tail ESS in the convergence criterion.
    # Previously converged only checked bulk ESS, leaving models with poor
    # tail sampling (inadequate exploration of tails, common with heavy-tailed
    # posteriors) incorrectly flagged as converged.
    converged = (max_rhat < 1.01) and (min_ess_bulk > 400) and (min_ess_tail > 400)

    return ConvergenceSummary(
        max_rhat=max_rhat,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        converged=converged,
        rhat_by_param=rhat_df,
        ess_by_param=rhat_df.select(["parameter", "ess_bulk"]),
        n_divergences=n_divergences,
    )


def compute_diagnostics(result: "BYM2Result") -> SpatialDiagnostics:
    """
    Compute the full diagnostics suite for a fitted BYM2Result.

    Includes convergence statistics and posterior summaries of the key
    hyperparameters (rho and sigma).  Moran's I on posterior residuals is
    not computed here - call moran_i() separately on your pre-fit residuals.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("diagnostics require arviz. uv add arviz") from exc

    conv = convergence_summary(result)

    def _param_summary(param_name: str) -> pl.DataFrame:
        samples = result.trace.posterior[param_name].values.ravel()
        return pl.DataFrame({
            "parameter": [param_name],
            "mean": [float(samples.mean())],
            "sd": [float(samples.std())],
            "q025": [float(np.quantile(samples, 0.025))],
            "q975": [float(np.quantile(samples, 0.975))],
        })

    rho_summary = _param_summary("rho")
    sigma_summary = _param_summary("sigma")

    return SpatialDiagnostics(
        convergence=conv,
        rho_summary=rho_summary,
        sigma_summary=sigma_summary,
        moran_post=None,
    )
