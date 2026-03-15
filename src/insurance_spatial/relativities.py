"""
Extract territory relativities from a fitted BYM2 model.

Relativities are multiplicative factors expressing how much each territory's
expected claim frequency differs from a baseline.  They are designed to slot
directly into an existing GLM as a log-scale offset column:

    ln_offset = ln(relativity) = b_i - reference_b

where reference_b is either:
  - the per-draw grand mean of b across all areas (default), giving a mean-centred
    set that multiplies to 1 in geometric terms, or
  - the per-draw b value for a specified base area (that area gets exactly 1.0
    for each posterior draw before the expectation is taken).

The per-draw subtraction is the key correctness requirement: subtracting a
point estimate (posterior mean) of the reference level instead of the per-draw
value underestimates credibility interval width, because it ignores uncertainty
in the reference level itself.  For areas with sparse data, this can make
intervals look materially tighter than they should be.

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
        Area identifier to use as the reference (relativity = 1.0 in expectation).
        If None, normalise to the per-draw geometric mean (log-scale grand mean = 0).
    credibility_interval :
        Width of the symmetric posterior interval.  0.95 gives 2.5%–97.5%.

    Returns
    -------
    pl.DataFrame with columns:
        area          str
        b_mean        float  posterior mean of b_i (log scale)
        b_sd          float  posterior SD of b_i (log scale)
        relativity    float  posterior mean of exp(b_i - reference_b_draw),
                             propagating uncertainty in the reference level
        lower         float  lower credibility bound on relativity
        upper         float  upper credibility bound on relativity
        ln_offset     float  b_mean - reference_b_mean, ready to use as GLM
                             log-offset (consistent with posterior mean on log scale)
    """
    trace = result.trace

    # Pull posterior samples of b: shape (chains, draws, N)
    b_samples = trace.posterior["b"].values  # (chains, draws, N)
    # Flatten to (total_draws, N)
    n_chains, n_draws, N = b_samples.shape
    b_flat = b_samples.reshape(n_chains * n_draws, N)  # (S, N)

    # Posterior mean and SD on log scale
    b_mean = b_flat.mean(axis=0)  # (N,)
    b_sd = b_flat.std(axis=0)     # (N,)

    # P0 fix: use per-draw reference level rather than a point estimate.
    # Subtracting the posterior mean of the reference area fixes it at its
    # expected value and ignores its uncertainty — credibility intervals on
    # all other relativities then don't account for uncertainty in the
    # denominator.  The correct approach subtracts the reference's draw-level
    # b from each draw, propagating that uncertainty into every interval.
    if base_area is not None:
        if base_area not in result.areas:
            raise ValueError(
                f"base_area '{base_area}' not found. "
                f"First few areas: {result.areas[:5]}"
            )
        ref_idx = result.areas.index(base_area)
        # Per-draw subtraction: shape (S, 1) broadcasts over (S, N)
        ref_b_draws = b_flat[:, ref_idx : ref_idx + 1]  # (S, 1)
        b_adjusted = b_flat - ref_b_draws                # (S, N)
        # Scalar reference for ln_offset (log-scale point estimate)
        reference_b_mean = b_mean[ref_idx]
    else:
        # Grand mean normalisation: subtract the per-draw mean across all areas
        # so that each draw individually sums to zero on log scale.
        b_adjusted = b_flat - b_flat.mean(axis=1, keepdims=True)  # (S, N)
        reference_b_mean = b_mean.mean()

    rel_samples = np.exp(b_adjusted)   # (S, N)

    alpha = 1.0 - credibility_interval
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    relativity_mean = rel_samples.mean(axis=0)
    lower_ci = np.quantile(rel_samples, lower_q, axis=0)
    upper_ci = np.quantile(rel_samples, upper_q, axis=0)

    # ln_offset: the log-scale relativity for direct use as a GLM offset.
    # This is the posterior mean of b_i minus the posterior mean of the
    # reference level.  We use the log-scale quantity (not log of the
    # multiplicative relativity) to avoid Jensen's inequality bias —
    # log(E[exp(X)]) > E[X] for non-degenerate X.
    ln_offset = b_mean - reference_b_mean

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
