"""
Extract territory relativities from a fitted BYM2 model.

Relativities are multiplicative factors expressing how much each territory's
expected claim frequency differs from a baseline.  They are designed to slot
directly into an existing GLM as a log-scale offset column:

    ln_offset = ln(relativity) = b_i - reference_b

where reference_b is either:
  - the grand mean of b across all areas (default), giving a mean-centred set
    that multiplies to 1 in geometric terms, or
  - the posterior mean of b for a specified base area (that area gets 1.0).

The output DataFrame is Polars and includes posterior mean, standard deviation,
and a symmetric credibility interval on the log scale and multiplicative scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from insurance_spatial.models import BYM2Result


def extract_relativities(
    result: "BYM2Result",
    base_area: Optional[str] = None,
    credibility_interval: float = 0.95,
) -> pl.DataFrame:
    """
    Extract territory relativities from a fitted BYM2Result.

    Parameters
    ----------
    result :
        Fitted BYM2Result object containing the MCMC trace.
    base_area :
        Area identifier to use as the reference (relativity = 1.0).
        If None, normalise to the geometric mean (log-scale grand mean = 0).
    credibility_interval :
        Width of the symmetric posterior interval.  0.95 gives 2.5%–97.5%.

    Returns
    -------
    pl.DataFrame with columns:
        area          str
        b_mean        float  posterior mean of b_i (log scale)
        b_sd          float  posterior SD of b_i (log scale)
        relativity    float  exp(b_i - reference_b), multiplicative factor
        lower         float  lower credibility bound on relativity
        upper         float  upper credibility bound on relativity
        ln_offset     float  = ln(relativity), ready to use as GLM log-offset
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("extract_relativities requires arviz. pip install arviz") from exc

    trace = result.trace

    # Pull posterior samples of b: shape (chains, draws, N)
    b_samples = trace.posterior["b"].values  # (chains, draws, N)
    # Flatten to (total_draws, N)
    n_chains, n_draws, N = b_samples.shape
    b_flat = b_samples.reshape(n_chains * n_draws, N)  # (S, N)

    # Posterior mean and SD on log scale
    b_mean = b_flat.mean(axis=0)  # (N,)
    b_sd = b_flat.std(axis=0)     # (N,)

    # Determine reference level
    if base_area is not None:
        if base_area not in result.areas:
            raise ValueError(
                f"base_area '{base_area}' not found. "
                f"First few areas: {result.areas[:5]}"
            )
        ref_idx = result.areas.index(base_area)
        # Reference is the posterior mean of the base area's b
        reference_b = b_mean[ref_idx]
    else:
        # Grand mean normalisation (geometric mean of relativities = 1)
        reference_b = b_mean.mean()

    # Compute relativities on the full posterior sample (preserves uncertainty)
    b_adjusted = b_flat - reference_b  # (S, N)
    rel_samples = np.exp(b_adjusted)   # (S, N)

    alpha = 1.0 - credibility_interval
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    relativity_mean = rel_samples.mean(axis=0)
    lower_ci = np.quantile(rel_samples, lower_q, axis=0)
    upper_ci = np.quantile(rel_samples, upper_q, axis=0)
    ln_offset = np.log(relativity_mean)

    df = pl.DataFrame(
        {
            "area": result.areas,
            "b_mean": b_mean.tolist(),
            "b_sd": b_sd.tolist(),
            "relativity": relativity_mean.tolist(),
            "lower": lower_ci.tolist(),
            "upper": upper_ci.tolist(),
            "ln_offset": ln_offset.tolist(),
        }
    ).with_columns(
        [
            pl.col("b_mean").cast(pl.Float64),
            pl.col("b_sd").cast(pl.Float64),
            pl.col("relativity").cast(pl.Float64),
            pl.col("lower").cast(pl.Float64),
            pl.col("upper").cast(pl.Float64),
            pl.col("ln_offset").cast(pl.Float64),
        ]
    )

    return df
