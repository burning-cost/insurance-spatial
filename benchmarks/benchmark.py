"""
Benchmark: insurance-spatial BYM2 spatial smoothing vs raw postcode experience
vs simple regional grouping for territory ratemaking.

The problem: postcode-level claim frequencies in personal lines are extremely
noisy. A postcode sector with 50 policy-years will show a zero frequency in
years with no claims and a 40% frequency in years with one. Neither is the
true underlying risk. Grouping postcodes into 5-10 large regions throws away
real geographic signal — the risk genuinely varies within a region.

BYM2 (Besag-York-Mollié 2) spatial smoothing solves this by fitting a prior
that borrows information from neighbouring postcodes. If nearby postcodes all
have high observed frequencies, the model treats a thin postcode's high rate
as signal, not noise. If an isolated thin postcode has a high rate with no
similar neighbours, the model discounts it.

The rho parameter directly tells you how much of the residual geographic variation
is spatially structured vs pure noise. A rho near 1 means the pattern is smooth
and BYM2 is doing real work.

Setup:
- 500 synthetic postcode sectors on a 20x25 grid (spatial autocorrelation in DGP)
- True log risk drawn from spatially smoothed noise (autocorrelation range ~5 postcodes)
- Exposure varies: ~100 thin postcodes have fewer than 30 policy-years
- Three approaches:
  (1) Raw experience: claims / exposure per postcode (noisy for thin areas)
  (2) Regional grouping: assign to 25 regions of 20 postcodes each, apply regional mean
  (3) BYM2: spatial smoothing using neighbour structure

Key metrics:
- MAE vs true risk by exposure tier
- Moran's I of residuals (before and after BYM2)
- Shrinkage effect on thin postcodes

Run:
    python benchmarks/benchmark.py

Install:
    pip install insurance-spatial pymc numpy scipy
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: insurance-spatial BYM2 smoothing vs raw vs regional grouping")
print("=" * 70)
print()

try:
    from insurance_spatial import (
        build_grid_adjacency,
        BYM2Model,
    )
    from insurance_spatial.diagnostics import moran_i
    print("insurance-spatial imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-spatial: {e}")
    print("Install with: pip install insurance-spatial")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data-generating process: spatially autocorrelated risk surface
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

NROWS, NCOLS = 20, 25
N_AREAS = NROWS * NCOLS  # 500 postcode sectors
TRUE_PORTFOLIO_FREQ = 0.07  # 7% claim frequency

print(f"DGP: {N_AREAS} postcode sectors on a {NROWS}x{NCOLS} grid")
print(f"     True portfolio claim frequency: {TRUE_PORTFOLIO_FREQ:.1%}")
print(f"     Spatially autocorrelated risk (smoothed noise, range ~5 postcodes)")
print()

# Build adjacency structure (queen contiguity)
adj = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")
print(f"Adjacency: {adj.n} areas, mean neighbours: {adj.neighbour_counts().mean():.1f}")
print(f"BYM2 scaling factor: {adj.scaling_factor:.4f}")
print()

# Generate spatially correlated log-risk via iterative spatial smoothing
# Row-normalised adjacency walks produce spatial autocorrelation
W_dense = adj.W.toarray().astype(np.float64)
row_sums = W_dense.sum(axis=1, keepdims=True)
W_rownorm = W_dense / np.maximum(row_sums, 1)

log_risk_noise = RNG.normal(0, 0.5, N_AREAS)
log_risk_smooth = log_risk_noise.copy()
for _ in range(3):
    log_risk_smooth = 0.6 * (W_rownorm @ log_risk_smooth) + 0.4 * log_risk_noise

# Centre and scale to portfolio mean
log_risk_smooth = log_risk_smooth - np.mean(log_risk_smooth) + np.log(TRUE_PORTFOLIO_FREQ)
true_rate = np.exp(log_risk_smooth)

# Exposure per postcode: heavy imbalance
log_exposure = RNG.normal(4.5, 1.2, N_AREAS)
exposure = np.exp(log_exposure)
exposure = np.clip(exposure, 5.0, 500.0)

# Observed claims: Poisson
claims = RNG.poisson(true_rate * exposure).astype(np.int64)
raw_rate = claims / exposure

print(f"Claims: {claims.sum():,} total, mean exposure: {exposure.mean():.0f} policy-years/area")

# Exposure tiers
thin_mask = exposure < 30
medium_mask = (exposure >= 30) & (exposure < 100)
thick_mask = exposure >= 100

print(f"Exposure tiers:")
print(f"  Thin   (< 30 py):  {thin_mask.sum():>4} areas")
print(f"  Medium (30-100 py): {medium_mask.sum():>3} areas")
print(f"  Thick  (> 100 py): {thick_mask.sum():>4} areas")
print()

# ---------------------------------------------------------------------------
# Baseline 1: Moran's I on raw residuals (confirm spatial autocorrelation exists)
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE 1: Moran's I on raw log(O/E) residuals")
print("-" * 70)
print()

portfolio_raw = claims.sum() / exposure.sum()
log_oe_raw = np.log(np.maximum(raw_rate / portfolio_raw, 1e-4))

moran_raw = moran_i(log_oe_raw, adj, n_permutations=499)
print(f"  Moran's I (raw):  I={moran_raw.statistic:.4f}, p={moran_raw.p_value:.4f}")
print(f"  Interpretation:   {moran_raw.interpretation}")
print()

# ---------------------------------------------------------------------------
# Baseline 2: Simple regional grouping (25 regions of 20 postcodes each)
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE 2: Regional grouping (25 regions, 20 postcodes each)")
print("-" * 70)
print()

N_REGIONS = 25
REGION_SIZE = N_AREAS // N_REGIONS

region_ids = np.minimum(np.arange(N_AREAS) // REGION_SIZE, N_REGIONS - 1)
regional_rate = np.zeros(N_AREAS)
for r in range(N_REGIONS):
    mask = region_ids == r
    reg_rate = claims[mask].sum() / exposure[mask].sum()
    regional_rate[mask] = reg_rate

mae_raw_thin = np.mean(np.abs(raw_rate[thin_mask] - true_rate[thin_mask]))
mae_raw_medium = np.mean(np.abs(raw_rate[medium_mask] - true_rate[medium_mask]))
mae_raw_thick = np.mean(np.abs(raw_rate[thick_mask] - true_rate[thick_mask]))

mae_regional_thin = np.mean(np.abs(regional_rate[thin_mask] - true_rate[thin_mask]))
mae_regional_medium = np.mean(np.abs(regional_rate[medium_mask] - true_rate[medium_mask]))
mae_regional_thick = np.mean(np.abs(regional_rate[thick_mask] - true_rate[thick_mask]))

print(f"  Portfolio raw rate: {portfolio_raw:.4f}")
print(f"  MAE (raw)      — thin: {mae_raw_thin:.4f}, medium: {mae_raw_medium:.4f}, thick: {mae_raw_thick:.4f}")
print(f"  MAE (regional) — thin: {mae_regional_thin:.4f}, medium: {mae_regional_medium:.4f}, thick: {mae_regional_thick:.4f}")
print()

# ---------------------------------------------------------------------------
# Library: BYM2 spatial smoothing
# ---------------------------------------------------------------------------

print("-" * 70)
print("LIBRARY: insurance-spatial BYM2 (spatial + IID random effects)")
print("-" * 70)
print()

bym2 = BYM2Model(
    adjacency=adj,
    draws=500,
    chains=2,
    tune=500,
    target_accept=0.9,
)

t0 = time.time()
result = bym2.fit(
    claims=claims,
    exposure=exposure,
    random_seed=42,
)
fit_time = time.time() - t0
print(f"  BYM2 fit time: {fit_time:.1f}s ({bym2.draws} draws x {bym2.chains} chains)")

# Extract territory relativities (normalised to geometric mean = 1.0)
rels = result.territory_relativities()
rels_pd = rels.to_pandas().sort_values("area")
bym2_relativity = rels_pd["relativity"].values

# Smoothed rates: relativity * portfolio raw mean
bym2_rate = bym2_relativity * portfolio_raw

mae_bym2_thin = np.mean(np.abs(bym2_rate[thin_mask] - true_rate[thin_mask]))
mae_bym2_medium = np.mean(np.abs(bym2_rate[medium_mask] - true_rate[medium_mask]))
mae_bym2_thick = np.mean(np.abs(bym2_rate[thick_mask] - true_rate[thick_mask]))

# Moran's I on BYM2 residuals
log_oe_bym2 = np.log(np.maximum(raw_rate / np.maximum(bym2_rate, 1e-8), 1e-4))
moran_post = moran_i(log_oe_bym2, adj, n_permutations=499)
print(f"\n  Moran's I (post BYM2): I={moran_post.statistic:.4f}, p={moran_post.p_value:.4f}")
print(f"  Interpretation: {moran_post.interpretation}")
print()

# Spatial fraction (rho)
diag = result.diagnostics()
print(f"  Spatial structure parameter rho (ICAR fraction vs IID noise):")
print(diag.rho_summary)
print()

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY: MAE vs true spatial risk by exposure tier")
print("=" * 70)
print()

print(f"  {'Tier':<10} {'n_areas':>8} {'Raw MAE':>10} {'Regional MAE':>13} {'BYM2 MAE':>10} {'Best':>8}")
print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*13} {'-'*10} {'-'*8}")

tier_info = [
    ("Thin", thin_mask, mae_raw_thin, mae_regional_thin, mae_bym2_thin),
    ("Medium", medium_mask, mae_raw_medium, mae_regional_medium, mae_bym2_medium),
    ("Thick", thick_mask, mae_raw_thick, mae_regional_thick, mae_bym2_thick),
]

for tier_name, mask, mae_r, mae_reg, mae_b in tier_info:
    if not mask.any():
        continue
    n = mask.sum()
    best_mae = min(mae_r, mae_reg, mae_b)
    best = "BYM2" if mae_b == best_mae else ("Regional" if mae_reg == best_mae else "Raw")
    print(f"  {tier_name:<10} {n:>8} {mae_r:>10.4f} {mae_reg:>13.4f} {mae_b:>10.4f} {best:>8}")

mae_r_all = np.mean(np.abs(raw_rate - true_rate))
mae_reg_all = np.mean(np.abs(regional_rate - true_rate))
mae_b_all = np.mean(np.abs(bym2_rate - true_rate))
print(f"  {'All':<10} {N_AREAS:>8} {mae_r_all:>10.4f} {mae_reg_all:>13.4f} {mae_b_all:>10.4f}")
print()

print("SPATIAL AUTOCORRELATION ABSORPTION")
print(f"  Pre-smoothing  Moran's I: {moran_raw.statistic:+.4f}  (p={moran_raw.p_value:.3f})  "
      f"{'SIGNIFICANT' if moran_raw.significant else 'not significant'}")
print(f"  Post-smoothing Moran's I: {moran_post.statistic:+.4f}  (p={moran_post.p_value:.3f})  "
      f"{'still significant' if moran_post.significant else 'absorbed by model'}")
print()

print("INTERPRETATION")
print(f"  Thin areas (<30 py): BYM2 MAE is lower than raw because the model")
print(f"  borrows from neighbouring postcodes via the ICAR prior. A thin")
print(f"  postcode in a high-risk cluster gets appropriately elevated rates;")
print(f"  an isolated spike gets discounted as noise.")
print()
print(f"  Regional grouping is a blunt instrument: all postcodes in the same")
print(f"  administrative group get the same rate, ignoring within-region variation.")
print(f"  This systematically underestimates risk variation at postcode level.")
print()
print(f"  BYM2 rho: if high, geographic risk is spatially structured (smooth).")
print(f"  If rho is near zero, variation is IID noise — spatial smoothing still")
print(f"  helps thin areas but the structured component is less important.")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
