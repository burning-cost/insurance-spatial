# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-spatial: BYM2 vs GLM with region dummies vs naive geographic mean
# MAGIC
# MAGIC This benchmark answers: does spatial smoothing via BYM2 add value over
# MAGIC the standard actuarial approach (a GLM with postcode-sector dummies)?
# MAGIC
# MAGIC Three methods compared:
# MAGIC 1. **Naive geographic mean** — one number per territory, no credibility
# MAGIC 2. **GLM with region dummies** — industry standard, unregularised
# MAGIC 3. **BYM2 spatial model** — borrows strength from neighbours, quantifies spatial vs noise
# MAGIC
# MAGIC The key test is held-out thin territories: sectors with <30 policy-years.
# MAGIC BYM2 should outperform on those while staying competitive on thick sectors.

# COMMAND ----------

# Install dependencies
# BYM2 requires PyMC; GLM baseline uses statsmodels.
# On Databricks Free Edition the cluster has scipy/numpy/sklearn pre-installed.
# PyMC and insurance-spatial are added here.

import subprocess
import sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *packages])

pip_install("insurance-spatial[mcmc]", "statsmodels")

print("Dependencies installed.")

# COMMAND ----------

import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
# 10x10 grid = 100 postcode sectors (manageable for Databricks Free Edition)
# True risk surface is an urban/rural gradient (centre = urban high-risk)
# plus cluster effects (a pocket of high-risk in one corner = excess flood zone).
# This mimics realistic UK geographic risk structure.

RNG = np.random.default_rng(42)
NROWS, NCOLS = 10, 10
N = NROWS * NCOLS  # 100 areas
TRUE_MEAN_FREQ = 0.08   # 8% base claim frequency (motor-like)

print("=" * 65)
print("insurance-spatial benchmark: BYM2 vs GLM dummies vs naive mean")
print("=" * 65)
print(f"\nDGP: {N} postcode sectors on a {NROWS}x{NCOLS} grid")
print(f"     True mean claim frequency: {TRUE_MEAN_FREQ:.1%}")
print(f"     Spatial structure: urban/rural gradient + cluster effects")

# Grid coordinates (row = north-south, col = east-west)
row_idx = np.array([i // NCOLS for i in range(N)])
col_idx = np.array([i % NCOLS for i in range(N)])

# Urban/rural gradient: centre of grid is urban (high-risk)
centre_r, centre_c = (NROWS - 1) / 2, (NCOLS - 1) / 2
dist_to_centre = np.sqrt((row_idx - centre_r) ** 2 + (col_idx - centre_c) ** 2)
max_dist = np.sqrt(centre_r ** 2 + centre_c ** 2)
urban_effect = 0.4 * (1.0 - dist_to_centre / max_dist)  # up to +0.4 log units urban

# Cluster effect: top-right corner has elevated risk (e.g. flood zone / crime hotspot)
cluster_effect = np.where(
    (row_idx <= 2) & (col_idx >= 7),
    RNG.normal(0.30, 0.05, N),
    0.0,
)

# IID noise (the part BYM2 should NOT smooth — genuine area-level idiosyncrasy)
iid_noise = RNG.normal(0, 0.15, N)

log_true = np.log(TRUE_MEAN_FREQ) + urban_effect + cluster_effect + iid_noise
log_true = log_true - log_true.mean() + np.log(TRUE_MEAN_FREQ)  # re-centre
true_rate = np.exp(log_true)

# Exposure: realistic heterogeneity
# Urban areas (near centre) have higher policy counts
base_exp = np.exp(RNG.normal(4.2, 0.9, N))
urban_exposure_boost = 2.0 * (1.0 - dist_to_centre / max_dist)
exposure = np.clip(base_exp * np.exp(urban_exposure_boost), 5.0, 500.0)

# Observed claims: Poisson
claims = RNG.poisson(true_rate * exposure).astype(np.int64)
raw_rate = np.where(exposure > 0, claims / exposure, 0.0)

# Thin vs thick territories
thin_mask = exposure < 30
thick_mask = exposure >= 100
held_out_mask = thin_mask  # test on thin territories — hardest case

print(f"\nData summary:")
print(f"  Total policies (exposure): {exposure.sum():,.0f} policy-years")
print(f"  Total observed claims:     {claims.sum():,}")
print(f"  Mean exposure per area:    {exposure.mean():.0f} policy-years")
print(f"  Thin areas (<30 py):       {thin_mask.sum()} of {N}")
print(f"  Thick areas (>=100 py):    {thick_mask.sum()} of {N}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Adjacency matrix and Moran's I: confirm spatial signal exists
# ---------------------------------------------------------------------------

from insurance_spatial import build_grid_adjacency
from insurance_spatial.diagnostics import moran_i

adj = build_grid_adjacency(NROWS, NCOLS, connectivity="rook")  # rook: 4-neighbours
print(f"Adjacency: {adj.n} areas, mean neighbours: {adj.neighbour_counts().mean():.1f}")
print(f"BYM2 scaling factor: {adj.scaling_factor:.4f}")

portfolio_mean = claims.sum() / exposure.sum()
log_oe_raw = np.log(np.maximum(raw_rate / portfolio_mean, 1e-6))

t0 = time.perf_counter()
moran_before = moran_i(log_oe_raw, adj, n_permutations=499)
t_moran = time.perf_counter() - t0

print(f"\nMoran's I (raw log O/E rates):")
print(f"  I = {moran_before.statistic:.4f}")
print(f"  p = {moran_before.p_value:.4f}")
print(f"  {moran_before.interpretation}")
print(f"  ({t_moran:.1f}s)")
print()
if moran_before.p_value < 0.05:
    print("  => Significant spatial autocorrelation. BYM2 smoothing is warranted.")
else:
    print("  => No significant spatial autocorrelation. Plain GLM may suffice.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Method 1: Naive geographic mean (one rate per territory, raw O/E)
# ---------------------------------------------------------------------------
# The minimum baseline. No credibility, no borrowing. Each territory gets
# its own observed rate. This is what you see in most basic rating systems.

print("=" * 65)
print("Method 1: Naive geographic mean (raw observed rate per territory)")
print("=" * 65)

# Naive: just raw_rate. Already computed above.
naive_rate = raw_rate.copy()

# Protect against zero-exposure areas (none in our DGP, but defensive)
naive_rate = np.where(exposure > 0, naive_rate, portfolio_mean)

mse_naive_all   = float(np.mean((naive_rate - true_rate) ** 2))
mse_naive_thin  = float(np.mean((naive_rate[thin_mask]  - true_rate[thin_mask])  ** 2)) if thin_mask.sum() > 0 else float("nan")
mse_naive_thick = float(np.mean((naive_rate[thick_mask] - true_rate[thick_mask]) ** 2)) if thick_mask.sum() > 0 else float("nan")

rmse_naive_all   = np.sqrt(mse_naive_all)
rmse_naive_thin  = np.sqrt(mse_naive_thin)
rmse_naive_thick = np.sqrt(mse_naive_thick)

print(f"  RMSE overall:           {rmse_naive_all:.5f}")
print(f"  RMSE thin areas:        {rmse_naive_thin:.5f}   (n={thin_mask.sum()}, the problem zones)")
print(f"  RMSE thick areas:       {rmse_naive_thick:.5f}   (n={thick_mask.sum()}, reliable)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Method 2: GLM with region dummies (industry standard)
# ---------------------------------------------------------------------------
# This is what most UK pricing teams actually do.
# OLS/Poisson GLM where each territory is a separate dummy variable.
# No regularisation, no borrowing from neighbours.
# Thin territories get noisy estimates; the model cannot share information.

import statsmodels.api as sm

print("=" * 65)
print("Method 2: Poisson GLM with region dummies (industry standard)")
print("=" * 65)

t0 = time.perf_counter()

# Design matrix: intercept + N-1 territory dummies (drop first for identification)
# Area 0 is the reference territory.
X_dummy = np.zeros((N, N), dtype=np.float64)
for i in range(N):
    X_dummy[i, i] = 1.0  # one dummy per area (redundant intercept handled by GLM)

# Add log(exposure) as offset
glm_poisson = sm.GLM(
    claims,
    X_dummy,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exposure, 1e-10)),
)
glm_result = glm_poisson.fit(method="irls", maxiter=100, disp=False)
t_glm = time.perf_counter() - t0

# Fitted rates: exp(eta) / exposure where eta = log(exposure) + X*beta
glm_fitted_counts = glm_result.fittedvalues
glm_rate = glm_fitted_counts / np.maximum(exposure, 1e-10)

mse_glm_all   = float(np.mean((glm_rate - true_rate) ** 2))
mse_glm_thin  = float(np.mean((glm_rate[thin_mask]  - true_rate[thin_mask])  ** 2)) if thin_mask.sum() > 0 else float("nan")
mse_glm_thick = float(np.mean((glm_rate[thick_mask] - true_rate[thick_mask]) ** 2)) if thick_mask.sum() > 0 else float("nan")

rmse_glm_all   = np.sqrt(mse_glm_all)
rmse_glm_thin  = np.sqrt(mse_glm_thin)
rmse_glm_thick = np.sqrt(mse_glm_thick)

# Moran's I on GLM residuals — should remain significant if spatial ignored
log_oe_glm = np.log(np.maximum(raw_rate / np.maximum(glm_rate, 1e-8), 1e-6))
moran_glm = moran_i(log_oe_glm, adj, n_permutations=299)

print(f"  GLM fit time:           {t_glm:.2f}s")
print(f"  RMSE overall:           {rmse_glm_all:.5f}")
print(f"  RMSE thin areas:        {rmse_glm_thin:.5f}   (n={thin_mask.sum()})")
print(f"  RMSE thick areas:       {rmse_glm_thick:.5f}   (n={thick_mask.sum()})")
print(f"  Moran's I (residuals):  I={moran_glm.statistic:.4f}, p={moran_glm.p_value:.4f}")
print(f"  {moran_glm.interpretation}")
print()
print("  Note: Poisson GLM with territory dummies = MLE rates per territory.")
print("  For thin areas, MLE is identical to naive mean — no pooling occurs.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Method 3: BYM2 spatial model (insurance-spatial)
# ---------------------------------------------------------------------------
# PyMC-based Bayesian model that:
# - Borrows strength from spatial neighbours via ICAR random effect
# - Learns rho: proportion of residual variance that is spatially structured
# - Applies more smoothing to thin territories (less data => prior dominates)
# - Leaves thick territories largely unchanged (data overrides prior)

from insurance_spatial import BYM2Model

print("=" * 65)
print("Method 3: BYM2 spatial smoothing (insurance-spatial)")
print("=" * 65)
print("  Fitting PyMC BYM2 model (MCMC: 2 chains x 600 draws)...")
print("  This takes 2-5 minutes on Databricks Free Edition. Be patient.")

bym2 = BYM2Model(
    adjacency=adj,
    draws=600,
    chains=2,
    tune=600,
    target_accept=0.9,
    progressbar=False,
)

t0 = time.perf_counter()
result = bym2.fit(claims=claims, exposure=exposure, random_seed=42)
t_bym2 = time.perf_counter() - t0

print(f"  Fit time: {t_bym2:.1f}s")

# Territory relativities: exp(b_i) for each area, relative to portfolio mean
rels = result.territory_relativities()
rels_pd = rels.to_pandas().sort_values("area").reset_index(drop=True)
bym2_rate = rels_pd["relativity"].values * portfolio_mean

mse_bym2_all   = float(np.mean((bym2_rate - true_rate) ** 2))
mse_bym2_thin  = float(np.mean((bym2_rate[thin_mask]  - true_rate[thin_mask])  ** 2)) if thin_mask.sum() > 0 else float("nan")
mse_bym2_thick = float(np.mean((bym2_rate[thick_mask] - true_rate[thick_mask]) ** 2)) if thick_mask.sum() > 0 else float("nan")

rmse_bym2_all   = np.sqrt(mse_bym2_all)
rmse_bym2_thin  = np.sqrt(mse_bym2_thin)
rmse_bym2_thick = np.sqrt(mse_bym2_thick)

log_oe_bym2 = np.log(np.maximum(raw_rate / np.maximum(bym2_rate, 1e-8), 1e-6))
moran_post = moran_i(log_oe_bym2, adj, n_permutations=299)

diag = result.diagnostics()
rho_summary = diag.rho_summary

print(f"  RMSE overall:           {rmse_bym2_all:.5f}")
print(f"  RMSE thin areas:        {rmse_bym2_thin:.5f}   (n={thin_mask.sum()})")
print(f"  RMSE thick areas:       {rmse_bym2_thick:.5f}   (n={thick_mask.sum()})")
print(f"  Moran's I (residuals):  I={moran_post.statistic:.4f}, p={moran_post.p_value:.4f}")
print(f"  {moran_post.interpretation}")
print()
print(f"  Spatial fraction rho: {rho_summary}")
print(f"  Max R-hat: {diag.convergence.max_rhat:.4f}  (want <1.01 for convergence)")
print(f"  Min ESS:   {diag.convergence.min_ess_bulk:.0f}  (want >400)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Summary table and interpretation
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 65)
print()

methods = ["Naive mean", "GLM dummies", "BYM2"]
rmse_all   = [rmse_naive_all,   rmse_glm_all,   rmse_bym2_all]
rmse_thin  = [rmse_naive_thin,  rmse_glm_thin,  rmse_bym2_thin]
rmse_thick = [rmse_naive_thick, rmse_glm_thick, rmse_bym2_thick]

print(f"  {'Method':<18} {'RMSE all':>10} {'RMSE thin':>10} {'RMSE thick':>11}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*11}")
for m, a, tn, tk in zip(methods, rmse_all, rmse_thin, rmse_thick):
    print(f"  {m:<18} {a:>10.5f} {tn:>10.5f} {tk:>11.5f}")

print()
print(f"  Moran's I (pre-fit raw O/E):  I={moran_before.statistic:.4f}, p={moran_before.p_value:.4f}")
print(f"  Moran's I (GLM residuals):    I={moran_glm.statistic:.4f},   p={moran_glm.p_value:.4f}")
print(f"  Moran's I (BYM2 residuals):   I={moran_post.statistic:.4f},   p={moran_post.p_value:.4f}")

# Thin-area improvement
improve_thin_vs_naive = (rmse_naive_thin - rmse_bym2_thin) / rmse_naive_thin * 100
improve_thin_vs_glm   = (rmse_glm_thin  - rmse_bym2_thin) / rmse_glm_thin  * 100
improve_all_vs_naive  = (rmse_naive_all  - rmse_bym2_all)  / rmse_naive_all  * 100

print()
print(f"  Thin-area RMSE improvement (BYM2 vs naive): {improve_thin_vs_naive:+.1f}%")
print(f"  Thin-area RMSE improvement (BYM2 vs GLM):   {improve_thin_vs_glm:+.1f}%")
print(f"  Overall RMSE improvement (BYM2 vs naive):   {improve_all_vs_naive:+.1f}%")

print()
print("INTERPRETATION")
print("-" * 65)
print(f"  Pre-fit Moran's I = {moran_before.statistic:.3f} (p={moran_before.p_value:.3f}): the raw rates")
print(f"  are spatially autocorrelated. Nearby territories share risk.")
print()
print(f"  Naive mean and GLM dummies produce identical thin-area estimates:")
print(f"  both apply no credibility pooling. With <30 policy-years, the")
print(f"  observed rate can swing by 100%+ around the true rate.")
print()
print(f"  BYM2 smooths thin-area estimates toward a weighted spatial average")
print(f"  of neighbours, controlled by rho (estimated proportion of variance")
print(f"  that is spatially structured). The stronger the spatial signal,")
print(f"  the more it borrows; if the signal is weak (rho near 0), it reverts")
print(f"  to simple shrinkage toward the global mean.")
print()
print(f"  Post-BYM2 Moran's I = {moran_post.statistic:.3f} (p={moran_post.p_value:.3f}):")
if moran_post.p_value < 0.05:
    print(f"  Some residual spatial structure remains. Consider adding covariates")
    print(f"  or using the two-stage pipeline (GLM O/E then BYM2 on residuals).")
else:
    print(f"  Residual spatial structure is not significant — BYM2 has absorbed")
    print(f"  the geographic variation in the data.")
print()
print(f"  Honest caveat: on thick territories (>=100py) the three methods")
print(f"  converge. BYM2 earns its complexity only on thin portfolios or")
print(f"  when the pricing team wants credibility intervals on territory factors.")

print(f"\nBenchmark complete.")
