# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: BYM2 spatial smoothing vs naive territory banding
# MAGIC
# MAGIC **Library:** `insurance-spatial` — BYM2 spatial territory ratemaking using PyMC 5 ICAR models.
# MAGIC Applies spatial smoothing to territory rate estimates, borrowing strength from neighbouring
# MAGIC areas to stabilise thin territories without discarding their local signal entirely.
# MAGIC
# MAGIC **Baselines:**
# MAGIC - *Flat territory bands* — group postcodes into 5 bands by raw observed frequency (quintiles).
# MAGIC   The crude but common approach: someone runs a cross-tab, draws lines on a map, and calls it
# MAGIC   a territory model. Looks robust; creates hard discontinuities and badly misestimates thin areas.
# MAGIC - *Grand mean* — single portfolio rate applied everywhere. Zero geographic discrimination.
# MAGIC   Included as the lower bound of usefulness.
# MAGIC
# MAGIC **Dataset:** Synthetic Poisson frequency data on a 12×12 grid of territories (144 areas).
# MAGIC Known DGP with genuine spatial autocorrelation. True rates vary smoothly across the grid
# MAGIC plus spatially correlated noise — the structure you expect from UK territory data where
# MAGIC deprivation, urbanisation, and theft risk all have spatial gradients.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC **Library version:** 0.2.0
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This notebook makes the case for spatial smoothing on the problem it exists to solve:
# MAGIC estimating territory relativities when geographic data has genuine spatial structure and
# MAGIC heterogeneous exposure (some territories are data-rich, most are thin).
# MAGIC
# MAGIC The core proposition: raw frequency aggregated to territory level is unbiased but noisy,
# MAGIC especially for thin territories. Grouping thin territories into bands based on their noisy
# MAGIC raw experience just aggregates the noise — the band edges are artefacts of sampling
# MAGIC variation, not genuine rate cliffs. BYM2 respects the spatial graph: it pulls estimates
# MAGIC for each territory toward its neighbours, with the degree of smoothing calibrated from
# MAGIC the data through the rho and sigma parameters.
# MAGIC
# MAGIC **Evaluation:** MSE against known DGP true rates. We report overall, thin territory,
# MAGIC and thick territory performance. We also run Moran's I on residuals before and after
# MAGIC fitting — the BYM2 model should absorb the spatial signal (Moran's I near zero on
# MAGIC posterior residuals), while the banding approach leaves significant autocorrelation behind.
# MAGIC
# MAGIC **PyMC note:** PyMC 5 requires `pymc` and `pytensor`. On Databricks serverless compute
# MAGIC these may not install cleanly without a specific `%pip install` order — see the setup
# MAGIC cell below. The MCMC run (500 draws × 2 chains, 500 tune) takes approximately 3–8 minutes
# MAGIC on a standard Databricks cluster. If you want faster iteration, reduce `draws` and `tune`
# MAGIC to 200 each — convergence diagnostics will degrade but the qualitative results hold.
# MAGIC
# MAGIC **Problem type:** Frequency estimation — claims / exposure, Poisson DGP, territory-level
# MAGIC technical rate estimation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# PyMC and pytensor must be installed before the other dependencies to avoid
# conflicts with numpy ABI on some Databricks runtime versions.
%pip install "pymc>=5.10" "pytensor>=2.18"

# COMMAND ----------

# Library under test and supporting dependencies
%pip install "insurance-spatial[mcmc]"
%pip install arviz matplotlib seaborn pandas numpy polars scipy

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

# Library under test
from insurance_spatial import (
    build_grid_adjacency,
    BYM2Model,
    MoranI,
    moran_i,
    extract_relativities,
)
from insurance_spatial.diagnostics import moran_i, convergence_summary

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded successfully.")

import insurance_spatial as _spatial_mod
print(f"insurance-spatial version: {_spatial_mod.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generation Process

# COMMAND ----------

# MAGIC %md
# MAGIC We build a 12×12 grid of territories. Each cell represents a postcode sector or small
# MAGIC geographic unit — the granularity at which UK personal lines territory models typically
# MAGIC operate for premium rating.
# MAGIC
# MAGIC **Spatial frequency surface:**
# MAGIC
# MAGIC The true log-rate for each territory is a sum of three components:
# MAGIC
# MAGIC 1. A deterministic smooth spatial trend — a combination of sinusoidal and linear terms
# MAGIC    in grid (x, y) coordinates. This represents the kind of large-scale geographic gradient
# MAGIC    you see in UK motor data: higher frequencies in dense urban areas, lower in rural areas.
# MAGIC
# MAGIC 2. A spatially correlated noise term drawn from a conditional autoregressive (CAR) process.
# MAGIC    This represents medium-range correlation: a high-risk neighbourhood spills into adjacent
# MAGIC    territories, but the correlation decays with distance. In practice this picks up things
# MAGIC    like local crime hotspots, road quality clusters, and concentration of young drivers.
# MAGIC
# MAGIC 3. IID noise — pure territory-level idiosyncratic variation. Some territories are genuinely
# MAGIC    unusual regardless of their neighbours.
# MAGIC
# MAGIC **Exposure:**
# MAGIC
# MAGIC Policy count varies across territories. Urban cores have high exposure (thick territories
# MAGIC that could support their own rate estimate) while semi-rural and rural fringes have very
# MAGIC little exposure (thin territories that need to borrow strength). We simulate this by
# MAGIC assigning exposure as a function of position on the grid plus lognormal noise — mimicking
# MAGIC the urban-rural gradient in a typical UK personal lines book.
# MAGIC
# MAGIC **Claims:**
# MAGIC
# MAGIC Drawn from Poisson(true_rate * exposure). No overdispersion — keeping it clean to make the
# MAGIC spatial structure the only source of complexity.

# COMMAND ----------

RNG_SEED    = 2026
NROWS       = 12
NCOLS       = 12
N_AREAS     = NROWS * NCOLS  # 144 territories
N_BANDS     = 5              # quintile bands for the banding baseline
PORTFOLIO_FREQ = 0.08        # ~8% claims frequency, UK motor personal lines territory level

rng = np.random.default_rng(RNG_SEED)

# ── Grid coordinates ────────────────────────────────────────────────────────
# Normalised to [0, 1] in each dimension
rows_grid = np.array([i // NCOLS for i in range(N_AREAS)])  # row index 0..11
cols_grid = np.array([i  % NCOLS for i in range(N_AREAS)])  # col index 0..11

x = cols_grid / (NCOLS - 1)  # [0, 1]
y = rows_grid / (NROWS - 1)  # [0, 1]

# ── True log-rate surface ────────────────────────────────────────────────────
# Smooth trend: combination of sinusoidal surface and a linear NW-SE gradient
# This creates a "hill" of high frequency in the urban north-east corner,
# tapering to low rates in the rural south-west — typical UK motor pattern.
smooth_trend = (
    0.60 * np.sin(np.pi * x) * np.cos(np.pi * y * 0.8)   # N-S gradient
    + 0.40 * (x - 0.5)                                     # E-W gradient
    - 0.20 * (y - 0.5)                                     # secondary N-S
)

# Spatially correlated noise via a simple CAR-like process.
# We generate white noise and then smooth it using a 2D Gaussian kernel
# applied on the grid — this creates the correct spatial correlation structure
# without needing to solve the full CAR system for the DGP.
white_noise = rng.standard_normal(N_AREAS).reshape(NROWS, NCOLS)

# Convolve with a 3×3 averaging kernel to introduce spatial correlation
from scipy.ndimage import gaussian_filter
spatially_correlated_noise = gaussian_filter(white_noise, sigma=2.0).ravel()
# Rescale to have SD ≈ 0.25 (meaningful spatial noise, but not overwhelming)
spatially_correlated_noise = 0.25 * spatially_correlated_noise / spatially_correlated_noise.std()

# IID territory-level noise: SD ≈ 0.10
iid_noise = 0.10 * rng.standard_normal(N_AREAS)

# True log-rate for each territory
log_portfolio_freq = np.log(PORTFOLIO_FREQ)
log_true_rate = log_portfolio_freq + smooth_trend + spatially_correlated_noise + iid_noise

# True rate (frequency) per territory
true_rate = np.exp(log_true_rate)

print(f"DGP: {N_AREAS} territories on a {NROWS}×{NCOLS} grid")
print(f"\nTrue rate distribution:")
print(f"  Min:    {true_rate.min():.5f}")
print(f"  Median: {np.median(true_rate):.5f}")
print(f"  Mean:   {true_rate.mean():.5f}")
print(f"  Max:    {true_rate.max():.5f}")
print(f"  CV:     {true_rate.std() / true_rate.mean():.3f}  (coefficient of variation)")
print(f"\nGrand mean log rate: {log_true_rate.mean():.4f}")
print(f"SD of true log rates: {log_true_rate.std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exposure: urban-rural gradient

# COMMAND ----------

# Exposure varies by position on the grid.
# North-east corner (high x, high y) is dense urban: high exposure.
# South-west corner (low x, low y) is rural: thin exposure.
# We add lognormal noise to avoid a perfectly deterministic pattern.

# Deterministic component: 0-1 urbanisation score based on position
urbanisation = 0.5 * x + 0.5 * y  # 0 = rural SW, 1 = urban NE

# Expected exposure per territory: 20 policies in rural, 800 in urban core
# (representing the range you might see across postcode sectors in a mid-size portfolio)
log_base_exposure = np.log(20) + urbanisation * (np.log(800) - np.log(20))
log_exposure_noise = 0.35 * rng.standard_normal(N_AREAS)
exposure = np.round(np.exp(log_base_exposure + log_exposure_noise)).astype(float)
exposure = np.maximum(exposure, 5.0)  # at least 5 policies per territory

# Claims from the Poisson DGP
claims = rng.poisson(lam=true_rate * exposure).astype(np.int64)
# Raw observed frequency per territory
obs_freq = claims / exposure

# Categorise territories: thin (<50 exposure) vs thick (≥50)
THIN_THRESHOLD = 50
is_thin = exposure < THIN_THRESHOLD
n_thin  = int(is_thin.sum())
n_thick = int((~is_thin).sum())

area_labels = [f"r{r:02d}c{c:02d}" for r in range(NROWS) for c in range(NCOLS)]

print(f"Exposure distribution:")
print(f"  Min:    {exposure.min():.0f} policies")
print(f"  Median: {np.median(exposure):.0f} policies")
print(f"  Mean:   {exposure.mean():.0f} policies")
print(f"  Max:    {exposure.max():.0f} policies")
print(f"  Total:  {exposure.sum():,.0f} policy-years")
print(f"\nThin territories (exposure < {THIN_THRESHOLD}):  {n_thin}  ({n_thin/N_AREAS*100:.1f}%)")
print(f"Thick territories (exposure ≥ {THIN_THRESHOLD}): {n_thick} ({n_thick/N_AREAS*100:.1f}%)")
print(f"\nTotal claims: {claims.sum():,}")
print(f"Observed portfolio frequency: {claims.sum() / exposure.sum():.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect spatial pattern

# COMMAND ----------

# Visualise the DGP: true rate surface and observed frequency surface side by side

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True rate surface
true_grid = true_rate.reshape(NROWS, NCOLS)
im0 = axes[0].imshow(true_grid, cmap="RdYlGn_r", origin="upper", interpolation="nearest")
axes[0].set_title("DGP: True claim frequency per territory", fontweight="bold")
axes[0].set_xlabel("East  →")
axes[0].set_ylabel("North  →")
plt.colorbar(im0, ax=axes[0], label="True frequency")

# Observed frequency surface
obs_grid = obs_freq.reshape(NROWS, NCOLS)
im1 = axes[1].imshow(obs_grid, cmap="RdYlGn_r", origin="upper", interpolation="nearest")
axes[1].set_title("Observed claim frequency\n(Poisson noise on top of true surface)", fontweight="bold")
axes[1].set_xlabel("East  →")
axes[1].set_ylabel("North  →")
plt.colorbar(im1, ax=axes[1], label="Observed frequency")

# Exposure surface
exp_grid = exposure.reshape(NROWS, NCOLS)
im2 = axes[2].imshow(exp_grid, cmap="Blues", origin="upper", interpolation="nearest")
axes[2].set_title("Exposure per territory\n(urban NE = thick; rural SW = thin)", fontweight="bold")
axes[2].set_xlabel("East  →")
axes[2].set_ylabel("North  →")
plt.colorbar(im2, ax=axes[2], label="Policies")

plt.suptitle(
    f"Synthetic DGP — {N_AREAS} territories, known true rates, spatially correlated noise",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("/tmp/benchmark_spatial_dgp.png", dpi=120, bbox_inches="tight")
plt.show()
print("DGP plot saved to /tmp/benchmark_spatial_dgp.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pre-fit Diagnostic: Moran's I on Raw Residuals

# COMMAND ----------

# MAGIC %md
# MAGIC Before fitting any model, we run Moran's I on the raw observed frequencies to confirm
# MAGIC that spatial autocorrelation is present and that BYM2 is warranted.
# MAGIC
# MAGIC Moran's I tests whether nearby territories have more similar values than you would
# MAGIC expect by chance. A positive, significant result means: yes, the geographic structure
# MAGIC is real — smoothing toward neighbours will improve estimates.
# MAGIC
# MAGIC A non-significant or near-zero result here would mean the spatial model offers no
# MAGIC advantage. That would be unusual for UK territory data, but it does happen for lines
# MAGIC where claims are truly idiosyncratic (some commercial lines with very large single risks).
# MAGIC
# MAGIC We use the log of the raw frequency as the input to Moran's I, clipped to avoid −∞
# MAGIC for zero-claim territories. Log-scale is preferred because it is the natural scale of
# MAGIC the Poisson log-link model.

# COMMAND ----------

# Build the adjacency structure — rook connectivity on the 12×12 grid
adj = build_grid_adjacency(NROWS, NCOLS, connectivity="rook")

print(f"Adjacency matrix: {adj.n} areas, {adj.W.nnz // 2} edges")
print(f"Connected components: {adj.n_components()}")
print(f"BYM2 scaling factor: {adj.scaling_factor:.5f}")
print(f"\nNeighbour count distribution:")
nc = adj.neighbour_counts()
for k in sorted(np.unique(nc)):
    print(f"  {k} neighbours: {(nc == k).sum()} areas")

# COMMAND ----------

# Moran's I on log raw frequency (pre-model)
# Clip zero-claim territories: log(0) = -inf. Use 0.5/exposure as a continuity
# correction — adds half a claim to empty territories.
log_obs_freq = np.log(np.maximum(obs_freq, 0.5 / exposure))

t0 = time.perf_counter()
moran_raw = moran_i(log_obs_freq, adj, n_permutations=999)
moran_raw_time = time.perf_counter() - t0

print("Moran's I on log(observed frequency) — pre-fit")
print("=" * 60)
print(f"  Statistic I:    {moran_raw.statistic:.4f}")
print(f"  Expected E[I]:  {moran_raw.expected:.4f}  (null: -1/(N-1))")
print(f"  Z-score:        {moran_raw.z_score:.2f}")
print(f"  p-value:        {moran_raw.p_value:.4f}")
print(f"  Significant:    {moran_raw.significant}")
print(f"  Permutations:   {moran_raw.n_permutations}")
print(f"  Time:           {moran_raw_time:.2f}s")
print()
print(f"  Interpretation: {moran_raw.interpretation}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline 1: Grand Mean

# COMMAND ----------

# MAGIC %md
# MAGIC The portfolio-average frequency applied uniformly to every territory. The zero-information
# MAGIC prior — appropriate for a brand-new territory with no history whatsoever, but obviously
# MAGIC wrong for any territory with data. Included as the lower bound of usefulness.

# COMMAND ----------

t0 = time.perf_counter()

grand_mean = float(claims.sum()) / float(exposure.sum())
pred_grand = np.full(N_AREAS, grand_mean)

baseline_grand_time = time.perf_counter() - t0

print(f"Grand mean (portfolio frequency): {grand_mean:.6f}")
print(f"Fit time: {baseline_grand_time*1000:.3f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Baseline 2: Flat Territory Bands

# COMMAND ----------

# MAGIC %md
# MAGIC This is the naive but common approach: rank territories by their raw observed frequency,
# MAGIC divide into N bands (quintiles here, so 5 bands), and assign each territory the mean
# MAGIC frequency of its band. The pricing team draws lines on a map at the band boundaries
# MAGIC and applies a loading or discount factor to each band.
# MAGIC
# MAGIC The problems with this approach:
# MAGIC
# MAGIC 1. **Thin territory noise propagates into band assignments.** A thin territory with
# MAGIC    10 policies and 3 claims (observed frequency 0.30) will be classified into band 5
# MAGIC    (highest risk) even if its true underlying frequency is 0.08. The Poisson noise
# MAGIC    on a tiny sample is being treated as signal.
# MAGIC
# MAGIC 2. **Band edges create hard discontinuities.** Two territories on either side of a band
# MAGIC    boundary — with nearly identical true rates — get different loadings. In practice,
# MAGIC    pricing teams fight about these boundary cases constantly.
# MAGIC
# MAGIC 3. **The within-band average blends genuinely different territories.** A band labelled
# MAGIC    "medium risk" might contain a mix of genuinely medium-risk urban territories and
# MAGIC    thin rural territories with lucky-low observed experience. The band rate is wrong for both.
# MAGIC
# MAGIC 4. **No uncertainty quantification.** There is no way to know which band assignments
# MAGIC    you should trust and which are noise-driven.
# MAGIC
# MAGIC We implement this exactly as a pricing team would: quintile bands on raw observed
# MAGIC frequency, no weighting by exposure. The band rate is the simple average frequency
# MAGIC within the band.

# COMMAND ----------

t0 = time.perf_counter()

# Quintile cut-points on raw observed frequency
band_quantiles = np.linspace(0, 100, N_BANDS + 1)  # [0, 20, 40, 60, 80, 100]
band_edges = np.percentile(obs_freq, band_quantiles)
# Ensure edges are unique (can happen when many territories have freq = 0)
band_edges = np.unique(band_edges)

# Assign each territory to a band (using digitize — returns 1-indexed band)
band_assignment = np.digitize(obs_freq, band_edges[1:-1])  # 0-indexed bands 0..4

# Band rate: exposure-weighted mean frequency within each band
# (Using exposure-weighted mean is better practice than simple mean, but we deliberately
# use unweighted here to match the "quick and dirty" approach a pricing team would use.)
n_actual_bands = int(band_assignment.max()) + 1
band_rates = np.array([
    obs_freq[band_assignment == b].mean() if (band_assignment == b).any() else grand_mean
    for b in range(n_actual_bands)
])

pred_bands = band_rates[band_assignment]

baseline_bands_time = time.perf_counter() - t0

print(f"Territory banding: {n_actual_bands} bands from {N_BANDS} quintile cuts")
print(f"\nBand summary:")
for b in range(n_actual_bands):
    mask = band_assignment == b
    n_terr = int(mask.sum())
    n_thin_in_band = int((mask & is_thin).sum())
    print(f"  Band {b+1}: {n_terr:3d} territories  "
          f"({n_thin_in_band} thin)  "
          f"rate = {band_rates[b]:.5f}  "
          f"[freq range: {obs_freq[mask].min():.4f}–{obs_freq[mask].max():.4f}]")

print(f"\nFit time: {baseline_bands_time*1000:.3f}ms")

# COMMAND ----------

# How many thin territories were misclassified into a band that doesn't match
# where their true rate would put them?
true_band_assignment = np.digitize(true_rate, band_edges[1:-1])

# Band assignment error: did the raw-frequency banding put a territory in the
# correct band (based on the true rate)?
band_correct = (band_assignment == true_band_assignment)
band_correct_thin  = band_correct[is_thin]
band_correct_thick = band_correct[~is_thin]

print("Band assignment accuracy vs ground truth:")
print(f"  Overall:  {band_correct.mean()*100:.1f}%  ({band_correct.sum()}/{N_AREAS} territories)")
print(f"  Thin:     {band_correct_thin.mean()*100:.1f}%  ({band_correct_thin.sum()}/{n_thin})")
print(f"  Thick:    {band_correct_thick.mean()*100:.1f}%  ({band_correct_thick.sum()}/{n_thick})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Library Model: BYM2 Spatial Smoothing

# COMMAND ----------

# MAGIC %md
# MAGIC The BYM2 model treats territory rates as draws from a spatial random field. The key
# MAGIC insight is that the spatial structure in the data is genuine — nearby territories share
# MAGIC risk drivers — so we should use that structure to shrink estimates toward spatially
# MAGIC local averages rather than toward the global portfolio mean.
# MAGIC
# MAGIC **Model structure:**
# MAGIC
# MAGIC     y_i ~ Poisson(mu_i)
# MAGIC     log(mu_i) = log(E_i) + alpha + b_i
# MAGIC
# MAGIC     b_i = sigma * (sqrt(rho / s) * phi_i + sqrt(1 - rho) * theta_i)
# MAGIC
# MAGIC     phi ~ ICAR(W)         spatial (structured) component
# MAGIC     theta ~ Normal(0, 1)  IID (unstructured) component
# MAGIC     sigma ~ HalfNormal(1) total SD of the spatial effect
# MAGIC     rho ~ Beta(0.5, 0.5)  proportion of variance that is spatially structured
# MAGIC
# MAGIC The rho parameter is the key diagnostic: if rho is near 1, nearly all the geographic
# MAGIC variation is spatially smooth. If rho is near 0, the variation is purely idiosyncratic.
# MAGIC For UK motor territory data, you typically see rho in the range 0.7–0.95.
# MAGIC
# MAGIC **MCMC settings:** We use 500 draws per chain with 2 chains and 500 tuning steps.
# MAGIC This is deliberately conservative for a benchmark — use 1000 draws and 4 chains
# MAGIC in production to get more reliable tail estimates. On Databricks, 2 chains can run
# MAGIC in parallel on a multi-core cluster.
# MAGIC
# MAGIC If nutpie is available, the library uses it automatically (2–5x faster than PyMC's
# MAGIC default NUTS). If not, it falls back to PyMC NUTS with a warning.

# COMMAND ----------

# Build the BYM2 model
bym2 = BYM2Model(
    adjacency=adj,
    draws=500,
    chains=2,
    target_accept=0.90,
    tune=500,
)

print(f"BYM2 model configured:")
print(f"  Areas:         {adj.n}")
print(f"  Draws:         {bym2.draws} per chain")
print(f"  Chains:        {bym2.chains}")
print(f"  Tune steps:    {bym2.tune}")
print(f"  Target accept: {bym2.target_accept}")
print(f"  Scaling factor s: {adj.scaling_factor:.5f}")

# COMMAND ----------

# Fit the model — this runs MCMC and will take a few minutes.
# Progress bars will appear in the cell output.
print("Fitting BYM2 model via MCMC...")
print(f"Start time: {datetime.utcnow().isoformat()}Z")

t0 = time.perf_counter()

bym2_result = bym2.fit(
    claims=claims,
    exposure=exposure,
    random_seed=RNG_SEED,
)

bym2_fit_time = time.perf_counter() - t0

print(f"\nFit complete.")
print(f"End time:  {datetime.utcnow().isoformat()}Z")
print(f"Fit time:  {bym2_fit_time:.1f}s  ({bym2_fit_time/60:.1f} min)")
print(f"Areas:     {bym2_result.n_areas}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convergence diagnostics

# COMMAND ----------

# MAGIC %md
# MAGIC Before using the model outputs, we verify that the MCMC chains have converged.
# MAGIC Two diagnostics:
# MAGIC
# MAGIC - **R-hat** (Gelman-Rubin): compares within-chain variance to between-chain variance.
# MAGIC   Values above 1.01 indicate the chains have not mixed properly and the estimates are
# MAGIC   unreliable. With only 2 chains and 500 draws this is a loose check — production
# MAGIC   runs should use 4+ chains.
# MAGIC
# MAGIC - **ESS** (effective sample size): accounts for autocorrelation within chains. ESS < 400
# MAGIC   means estimates of quantiles and intervals will be unstable. The bulk ESS governs the
# MAGIC   precision of central estimates; the tail ESS governs the precision of quantile estimates.
# MAGIC
# MAGIC **Divergent transitions:** If you see divergences (> 0), the sampler encountered regions
# MAGIC of posterior geometry it could not explore correctly. Increase `target_accept` to 0.95
# MAGIC and `tune` to 1000. For ICAR models, a small number of divergences (< 10 out of
# MAGIC 500*2=1000 samples) is usually not a problem, but investigate before publishing results.

# COMMAND ----------

conv = convergence_summary(bym2_result)

print("MCMC convergence diagnostics")
print("=" * 55)
print(f"  Converged (R-hat < 1.01, ESS > 400): {conv.converged}")
print(f"  Max R-hat:        {conv.max_rhat:.4f}  (threshold: < 1.01)")
print(f"  Min bulk ESS:     {conv.min_ess_bulk:.0f}  (threshold: > 400)")
print(f"  Min tail ESS:     {conv.min_ess_tail:.0f}")
print(f"  Divergent trans:  {conv.n_divergences}")
print()
print("Per-parameter R-hat and ESS:")
print(conv.rhat_by_param.to_pandas().to_string(index=False))

if not conv.converged:
    print()
    print("WARNING: chains have not fully converged.")
    print("Results below are indicative but should not be used for production.")
    print("Increase draws to 1000, tune to 1000, chains to 4.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Posterior hyperparameters

# COMMAND ----------

# Summarise the key hyperparameters: rho (spatial proportion) and sigma (total SD)
diag = bym2_result.diagnostics()

print("Posterior of rho (proportion of spatial variance that is structured):")
print(diag.rho_summary.to_pandas().to_string(index=False))
print()
print("Posterior of sigma (total SD of territory spatial effect):")
print(diag.sigma_summary.to_pandas().to_string(index=False))

rho_mean  = float(diag.rho_summary["mean"][0])
sigma_mean = float(diag.sigma_summary["mean"][0])

print()
print(f"Interpretation:")
print(f"  rho = {rho_mean:.3f} — {rho_mean*100:.0f}% of geographic variation is spatially structured.")
if rho_mean > 0.7:
    print(f"  High rho: neighbouring territories have highly correlated true rates.")
    print(f"  Spatial smoothing will have a strong effect, especially for thin territories.")
elif rho_mean > 0.3:
    print(f"  Moderate rho: spatial structure present but substantial IID noise too.")
else:
    print(f"  Low rho: geographic variation is mostly idiosyncratic (unusual for territory data).")
print(f"  sigma = {sigma_mean:.3f} — total SD of territory log-rate effects around the portfolio mean.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract territory relativities

# COMMAND ----------

# Extract posterior mean territory rates.
# We get the spatial effect b_i on the log scale; the estimated rate for territory i is:
#     exp(alpha_hat + b_i) * (1/E_i) ... but for benchmarking we want the rate directly.
#
# The territory_relativities() method returns exp(b_i - reference_b).
# To get estimated rates, we multiply the relativities by the portfolio mean alpha term.

rels = bym2_result.territory_relativities()  # normalised to geometric mean = 1

# Recover the portfolio-level intercept alpha from the trace
alpha_samples = bym2_result.trace.posterior["alpha"].values.ravel()
alpha_mean = float(alpha_samples.mean())

# Estimated rate per territory = grand_mean * relativity
# (Since log(mu_i) = log(E_i) + alpha + b_i, and E[rate] = E[mu/E] = exp(alpha) * exp(b_i)
# normalised to geometric mean, we have:
#   est_rate_i = exp(alpha) * relativity_i  (approximately, for posterior mean)
# However, the cleaner approach is to use the posterior mean of mu_i / E_i directly.)

mu_samples = bym2_result.trace.posterior["mu"].values  # (chains, draws, N)
n_chains, n_draws, N_areas = mu_samples.shape
mu_flat = mu_samples.reshape(n_chains * n_draws, N_areas)

# Posterior mean rate per territory
mu_mean_per_area = mu_flat.mean(axis=0)  # (N,)
pred_bym2 = mu_mean_per_area / exposure  # estimated frequency per territory

print(f"Estimated rates from BYM2:")
print(f"  Min:    {pred_bym2.min():.5f}")
print(f"  Median: {np.median(pred_bym2):.5f}")
print(f"  Mean:   {pred_bym2.mean():.5f}")
print(f"  Max:    {pred_bym2.max():.5f}")
print(f"\nPosterior mean alpha: {alpha_mean:.4f}  (exp(alpha) = {np.exp(alpha_mean):.5f})")

# Posterior SD of territory rates (uncertainty per territory)
mu_sd_per_area = mu_flat.std(axis=0) / exposure

print(f"\nPosterior SD of estimated rates (uncertainty):")
print(f"  Min:    {mu_sd_per_area.min():.5f}")
print(f"  Median: {np.median(mu_sd_per_area):.5f}")
print(f"  Max:    {mu_sd_per_area.max():.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Post-fit Diagnostic: Moran's I on Residuals

# COMMAND ----------

# MAGIC %md
# MAGIC This is the decisive test. We compute Moran's I on the residuals from each model:
# MAGIC
# MAGIC     residual_i = log(observed_rate_i) - log(estimated_rate_i)
# MAGIC
# MAGIC For a well-fitted spatial model, the residuals should be spatially random — there
# MAGIC should be no remaining autocorrelation for the model to exploit. Moran's I on the
# MAGIC BYM2 residuals should be near zero and non-significant.
# MAGIC
# MAGIC For the banding baseline and the grand mean, significant positive Moran's I on residuals
# MAGIC means the models have left real geographic signal on the table.

# COMMAND ----------

# Clip to avoid log(0)
floor_rate = 0.5 / exposure  # continuity correction for zero-claim territories

log_obs   = np.log(np.maximum(obs_freq, floor_rate))
log_grand = np.log(np.maximum(pred_grand, floor_rate))
log_bands = np.log(np.maximum(pred_bands, floor_rate))
log_bym2  = np.log(np.maximum(pred_bym2,  floor_rate))

resid_grand = log_obs - log_grand
resid_bands = log_obs - log_bands
resid_bym2  = log_obs - log_bym2

t0 = time.perf_counter()
moran_grand = moran_i(resid_grand, adj, n_permutations=999)
moran_bands = moran_i(resid_bands, adj, n_permutations=999)
moran_bym2  = moran_i(resid_bym2,  adj, n_permutations=999)
moran_post_time = time.perf_counter() - t0

print("Moran's I on model residuals (log O/E by territory)")
print("=" * 70)
print()

for label, m in [("Grand mean", moran_grand), ("Territory bands", moran_bands), ("BYM2", moran_bym2)]:
    sig_str = "SIGNIFICANT" if m.significant else "not significant"
    print(f"{label}:")
    print(f"  I = {m.statistic:.4f}   z = {m.z_score:.2f}   p = {m.p_value:.4f}   ({sig_str})")
    print(f"  {m.interpretation}")
    print()

print(f"Moran's I computation time (3 models): {moran_post_time:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Metrics: MSE vs Known True Rates

# COMMAND ----------

# MAGIC %md
# MAGIC We evaluate all three models against the known DGP true rates. This is only possible
# MAGIC because we generated the data ourselves — in production you never know the ground truth.
# MAGIC That is why synthetic benchmarks are essential: they are the only honest way to measure
# MAGIC territory rate estimation accuracy.
# MAGIC
# MAGIC **MSE** against true rates is the primary metric. We also report RMSE in frequency units
# MAGIC (more interpretable: an RMSE of 0.005 means estimates are off by about 0.5 percentage
# MAGIC points in frequency) and the mean absolute error.
# MAGIC
# MAGIC We report three subsets:
# MAGIC - **All territories** — the overall picture
# MAGIC - **Thin territories** (exposure < 50 policies) — where spatial borrowing earns its keep
# MAGIC - **Thick territories** (exposure ≥ 50 policies) — where raw data is already reliable

# COMMAND ----------

def mse(true, pred):
    return float(np.mean((np.asarray(true) - np.asarray(pred)) ** 2))

def rmse(true, pred):
    return float(np.sqrt(mse(true, pred)))

def mae(true, pred):
    return float(np.mean(np.abs(np.asarray(true) - np.asarray(pred))))

def pct_vs(reference_val, method_val):
    """Signed % change from reference. Negative = improvement over reference."""
    if reference_val == 0:
        return float("nan")
    return (method_val - reference_val) / reference_val * 100

# All territories
mse_grand_all = mse(true_rate, pred_grand)
mse_bands_all = mse(true_rate, pred_bands)
mse_bym2_all  = mse(true_rate, pred_bym2)

# Thin territories
mse_grand_thin = mse(true_rate[is_thin], pred_grand[is_thin])
mse_bands_thin = mse(true_rate[is_thin], pred_bands[is_thin])
mse_bym2_thin  = mse(true_rate[is_thin], pred_bym2[is_thin])

# Thick territories
mse_grand_thick = mse(true_rate[~is_thin], pred_grand[~is_thin])
mse_bands_thick = mse(true_rate[~is_thin], pred_bands[~is_thin])
mse_bym2_thick  = mse(true_rate[~is_thin], pred_bym2[~is_thin])

def winner(g, b, m):
    vals = {"Grand mean": g, "Bands": b, "BYM2": m}
    return min(vals, key=vals.get)

rows_metrics = [
    {
        "Subset":         f"All territories (n={N_AREAS})",
        "Grand mean MSE": f"{mse_grand_all:.2e}",
        "Bands MSE":      f"{mse_bands_all:.2e}",
        "BYM2 MSE":       f"{mse_bym2_all:.2e}",
        "BYM2 vs Grand":  f"{pct_vs(mse_grand_all, mse_bym2_all):+.1f}%",
        "BYM2 vs Bands":  f"{pct_vs(mse_bands_all, mse_bym2_all):+.1f}%",
        "Winner":         winner(mse_grand_all, mse_bands_all, mse_bym2_all),
    },
    {
        "Subset":         f"Thin (n={n_thin}, exp < {THIN_THRESHOLD})",
        "Grand mean MSE": f"{mse_grand_thin:.2e}",
        "Bands MSE":      f"{mse_bands_thin:.2e}",
        "BYM2 MSE":       f"{mse_bym2_thin:.2e}",
        "BYM2 vs Grand":  f"{pct_vs(mse_grand_thin, mse_bym2_thin):+.1f}%",
        "BYM2 vs Bands":  f"{pct_vs(mse_bands_thin, mse_bym2_thin):+.1f}%",
        "Winner":         winner(mse_grand_thin, mse_bands_thin, mse_bym2_thin),
    },
    {
        "Subset":         f"Thick (n={n_thick}, exp ≥ {THIN_THRESHOLD})",
        "Grand mean MSE": f"{mse_grand_thick:.2e}",
        "Bands MSE":      f"{mse_bands_thick:.2e}",
        "BYM2 MSE":       f"{mse_bym2_thick:.2e}",
        "BYM2 vs Grand":  f"{pct_vs(mse_grand_thick, mse_bym2_thick):+.1f}%",
        "BYM2 vs Bands":  f"{pct_vs(mse_bands_thick, mse_bym2_thick):+.1f}%",
        "Winner":         winner(mse_grand_thick, mse_bands_thick, mse_bym2_thick),
    },
]

metrics_df = pd.DataFrame(rows_metrics)

print("MSE against known DGP true territory rates")
print("=" * 100)
print(metrics_df.to_string(index=False))

print("\nRMSE (frequency units — more interpretable):")
for label, g, b, m in [
    ("All  ", mse_grand_all,   mse_bands_all,   mse_bym2_all),
    ("Thin ", mse_grand_thin,  mse_bands_thin,  mse_bym2_thin),
    ("Thick", mse_grand_thick, mse_bands_thick, mse_bym2_thick),
]:
    print(f"  {label}: grand={np.sqrt(g):.5f}  bands={np.sqrt(b):.5f}  bym2={np.sqrt(m):.5f}")

print("\nMAE:")
for label, mask in [("All  ", np.ones(N_AREAS, dtype=bool)), ("Thin ", is_thin), ("Thick", ~is_thin)]:
    print(f"  {label}: grand={mae(true_rate[mask], pred_grand[mask]):.5f}  "
          f"bands={mae(true_rate[mask], pred_bands[mask]):.5f}  "
          f"bym2={mae(true_rate[mask], pred_bym2[mask]):.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Thin Territory Deep Dive

# COMMAND ----------

# MAGIC %md
# MAGIC The thin territories are where the methods diverge most sharply. For a territory with
# MAGIC 20 policies and 2 claims (rate = 0.10), the observed frequency has a standard error of
# MAGIC approximately sqrt(0.10 / 20) ≈ 0.07 — nearly as large as the rate itself.
# MAGIC
# MAGIC The banding approach treats this noisy rate as signal and assigns the territory to a
# MAGIC band accordingly. BYM2 recognises the uncertainty (through the Poisson likelihood) and
# MAGIC shrinks the estimate toward the weighted average of the territory's neighbours.
# MAGIC
# MAGIC We look at the 10 thinnest territories individually to see what each method does.

# COMMAND ----------

# Sort thin territories by exposure (thinnest first)
thin_idx = np.where(is_thin)[0]
thin_sorted = thin_idx[np.argsort(exposure[thin_idx])][:20]  # up to 20 thinnest

thin_detail = pd.DataFrame({
    "area":      [area_labels[i] for i in thin_sorted],
    "exposure":  exposure[thin_sorted].astype(int),
    "claims":    claims[thin_sorted],
    "obs_freq":  np.round(obs_freq[thin_sorted], 5),
    "true_rate": np.round(true_rate[thin_sorted], 5),
    "pred_grand":np.round(pred_grand[thin_sorted], 5),
    "pred_bands":np.round(pred_bands[thin_sorted], 5),
    "pred_bym2": np.round(pred_bym2[thin_sorted], 5),
    "band_no":   band_assignment[thin_sorted] + 1,
})

# Absolute errors
thin_detail["err_grand"] = np.abs(thin_detail["pred_grand"] - thin_detail["true_rate"]).round(5)
thin_detail["err_bands"] = np.abs(thin_detail["pred_bands"] - thin_detail["true_rate"]).round(5)
thin_detail["err_bym2"]  = np.abs(thin_detail["pred_bym2"]  - thin_detail["true_rate"]).round(5)

# Which method is closest to truth for each territory?
thin_detail["winner"] = thin_detail.apply(
    lambda r: "grand" if r["err_grand"] <= r["err_bands"] and r["err_grand"] <= r["err_bym2"]
              else ("bands" if r["err_bands"] <= r["err_bym2"] else "bym2"),
    axis=1
)

print("20 thinnest territories: true rate vs estimated rates")
print(thin_detail.to_string(index=False))

print(f"\nWinner counts across {len(thin_sorted)} thinnest territories:")
print(thin_detail["winner"].value_counts().to_string())

# COMMAND ----------

# How often is the banding assignment for thin territories driven by noise?
# A thin territory's band is "noise-driven" if the observed frequency puts it
# in a different band than the true rate would.
thin_band_correct = (band_assignment[is_thin] == true_band_assignment[is_thin])
print(f"Thin territory band assignment accuracy: {thin_band_correct.mean()*100:.1f}%")
print(f"({thin_band_correct.sum()}/{n_thin} thin territories assigned to the correct band)")
print()
print("This is the fundamental problem with banding thin areas: the band assignment")
print("itself is unreliable because it is based on noisy raw experience.")
print()

# Compare the magnitude of noise-driven band jumps
band_jump = np.abs(band_assignment[is_thin].astype(int) - true_band_assignment[is_thin].astype(int))
print(f"Band jump distribution (|assigned band − correct band|) for thin territories:")
for j in sorted(np.unique(band_jump)):
    print(f"  Jump of {j}: {(band_jump == j).sum()} territories ({(band_jump == j).mean()*100:.0f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Visualisation

# COMMAND ----------

# MAGIC %md
# MAGIC Five plots:
# MAGIC
# MAGIC 1. **True vs estimated scatter** — one point per territory, all three methods.
# MAGIC    BYM2 points should cluster tighter around the 45-degree line.
# MAGIC
# MAGIC 2. **Estimated rate map** — BYM2 surface vs banding surface on the grid.
# MAGIC    The banding surface should look blocky; BYM2 should look smooth.
# MAGIC
# MAGIC 3. **MSE bar chart by model and segment type** — overall summary.
# MAGIC
# MAGIC 4. **Moran's I before and after** — shows that BYM2 absorbs the spatial signal.
# MAGIC
# MAGIC 5. **Thin territory error comparison** — direct comparison on the hardest cases.

# COMMAND ----------

fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

ax_scatter = fig.add_subplot(gs[0, 0])       # true vs estimated scatter
ax_map_bands = fig.add_subplot(gs[0, 1])     # banding rate map
ax_map_bym2 = fig.add_subplot(gs[0, 2])      # BYM2 rate map
ax_mse = fig.add_subplot(gs[1, 0])           # MSE bar chart
ax_moran = fig.add_subplot(gs[1, 1])         # Moran's I comparison
ax_moran_bands_map = fig.add_subplot(gs[1, 2])  # residual map (banding)
ax_thin = fig.add_subplot(gs[2, :])          # thin territory deep dive

THIN_COLOUR  = "#d62728"
THICK_COLOUR = "#1f77b4"
BYM2_COLOUR  = "#2ca02c"
BANDS_COLOUR = "#ff7f0e"
GRAND_COLOUR = "#9467bd"

# ── Plot 1: True vs estimated scatter ─────────────────────────────────────
colours = [THIN_COLOUR if t else THICK_COLOUR for t in is_thin]

all_vals = np.concatenate([true_rate, pred_grand, pred_bands, pred_bym2])
lim_lo = all_vals.min() * 0.90
lim_hi = all_vals.max() * 1.05

ax_scatter.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.2, alpha=0.5, label="Perfect (y = x)")
ax_scatter.scatter(true_rate, pred_grand, c=colours, marker="x", s=25, alpha=0.5, linewidths=1.2)
ax_scatter.scatter(true_rate, pred_bands, c=colours, marker="^", s=25, alpha=0.5)
ax_scatter.scatter(true_rate, pred_bym2,  c=colours, marker="o", s=30, alpha=0.80)

legend_elements = [
    Line2D([0],[0], color=GRAND_COLOUR, marker="x", linestyle="None", label="Grand mean"),
    Line2D([0],[0], color=BANDS_COLOUR, marker="^", linestyle="None", label="Bands"),
    Line2D([0],[0], color=BYM2_COLOUR,  marker="o", linestyle="None", label="BYM2"),
    mpatches.Patch(color=THIN_COLOUR,  label=f"Thin (exp < {THIN_THRESHOLD})"),
    mpatches.Patch(color=THICK_COLOUR, label="Thick"),
    Line2D([0],[0], color="black", linestyle="--", label="y = x"),
]
ax_scatter.legend(handles=legend_elements, fontsize=7, loc="upper left")
ax_scatter.set_xlabel("True rate (DGP)")
ax_scatter.set_ylabel("Estimated rate")
ax_scatter.set_title("True vs Estimated Rate\n(per territory)", fontweight="bold")
ax_scatter.set_xlim(lim_lo, lim_hi)
ax_scatter.set_ylim(lim_lo, lim_hi)
ax_scatter.grid(True, alpha=0.25)

# ── Plot 2: Banding rate map ───────────────────────────────────────────────
bands_grid = pred_bands.reshape(NROWS, NCOLS)
vmin = min(true_rate.min(), pred_bym2.min(), pred_bands.min()) * 0.95
vmax = max(true_rate.max(), pred_bym2.max(), pred_bands.max()) * 1.05
im_bands = ax_map_bands.imshow(bands_grid, cmap="RdYlGn_r", origin="upper",
                                interpolation="nearest", vmin=vmin, vmax=vmax)
ax_map_bands.set_title(f"Territory Bands estimate\n({n_actual_bands} quintile bands — blocky discontinuities)",
                        fontweight="bold")
ax_map_bands.set_xlabel("East  →")
ax_map_bands.set_ylabel("North  →")
plt.colorbar(im_bands, ax=ax_map_bands, label="Estimated frequency")

# ── Plot 3: BYM2 rate map ──────────────────────────────────────────────────
bym2_grid = pred_bym2.reshape(NROWS, NCOLS)
im_bym2 = ax_map_bym2.imshow(bym2_grid, cmap="RdYlGn_r", origin="upper",
                               interpolation="nearest", vmin=vmin, vmax=vmax)
ax_map_bym2.set_title("BYM2 estimate\n(spatially smoothed — respects neighbourhood structure)",
                       fontweight="bold")
ax_map_bym2.set_xlabel("East  →")
ax_map_bym2.set_ylabel("North  →")
plt.colorbar(im_bym2, ax=ax_map_bym2, label="Estimated frequency")

# ── Plot 4: MSE bar chart ──────────────────────────────────────────────────
subsets = ["All\n(n=144)", f"Thin\n(n={n_thin})", f"Thick\n(n={n_thick})"]
mse_g = [mse_grand_all, mse_grand_thin, mse_grand_thick]
mse_b = [mse_bands_all, mse_bands_thin, mse_bands_thick]
mse_m = [mse_bym2_all,  mse_bym2_thin,  mse_bym2_thick]

x3 = np.arange(len(subsets))
w3 = 0.26
ax_mse.bar(x3 - w3, mse_g, w3, label="Grand mean", color=GRAND_COLOUR, alpha=0.80)
ax_mse.bar(x3,      mse_b, w3, label="Bands",      color=BANDS_COLOUR, alpha=0.80)
ax_mse.bar(x3 + w3, mse_m, w3, label="BYM2",       color=BYM2_COLOUR,  alpha=0.85)
ax_mse.set_xticks(x3)
ax_mse.set_xticklabels(subsets, fontsize=9)
ax_mse.set_ylabel("MSE vs true DGP rates")
ax_mse.set_title("MSE by Model and Territory Type\n(lower is better)", fontweight="bold")
ax_mse.legend(fontsize=8)
ax_mse.grid(True, alpha=0.25, axis="y")

# ── Plot 5: Moran's I comparison ───────────────────────────────────────────
models_moran = ["Pre-fit\n(raw obs)", "Grand mean\nresiduals", "Bands\nresiduals", "BYM2\nresiduals"]
moran_stats  = [moran_raw.statistic, moran_grand.statistic, moran_bands.statistic, moran_bym2.statistic]
moran_sig    = [moran_raw.significant, moran_grand.significant, moran_bands.significant, moran_bym2.significant]
moran_cols   = ["#aec7e8" if not s else "#d62728" for s in moran_sig]
moran_cols[0] = "#aec7e8"  # pre-fit is always "before" — colour it neutral
if moran_raw.significant:
    moran_cols[0] = "#d62728"

bar_moran = ax_moran.bar(range(4), moran_stats, color=moran_cols, alpha=0.85, edgecolor="white")
ax_moran.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
ax_moran.axhline(moran_raw.expected, color="grey", linewidth=1.2, linestyle="--",
                  alpha=0.7, label=f"E[I] = {moran_raw.expected:.4f} (null)")

for i, (stat, sig) in enumerate(zip(moran_stats, moran_sig)):
    sig_label = "sig." if sig else "n.s."
    ax_moran.text(i, stat + 0.002, f"{stat:.3f}\n({sig_label})",
                   ha="center", va="bottom", fontsize=8)

ax_moran.set_xticks(range(4))
ax_moran.set_xticklabels(models_moran, fontsize=8)
ax_moran.set_ylabel("Moran's I")
ax_moran.set_title("Moran's I on Residuals\n(red = significant autocorrelation remaining)",
                    fontweight="bold")
ax_moran.legend(fontsize=8)
ax_moran.grid(True, alpha=0.25, axis="y")

sig_patch = mpatches.Patch(color="#d62728", label="Significant (p < 0.05)")
ns_patch  = mpatches.Patch(color="#aec7e8", label="Not significant")
ax_moran.legend(handles=[sig_patch, ns_patch,
                          Line2D([0],[0], color="grey", linestyle="--", label=f"E[I] (null)")],
                fontsize=7)

# ── Plot 6: BYM2 residual map ──────────────────────────────────────────────
bym2_resid_grid = resid_bym2.reshape(NROWS, NCOLS)
vabs = max(abs(resid_bym2).max(), 0.01)
im_resid = ax_moran_bands_map.imshow(bym2_resid_grid, cmap="RdBu", origin="upper",
                                      interpolation="nearest", vmin=-vabs, vmax=vabs)
ax_moran_bands_map.set_title("BYM2 log residuals (log O/E)\n(should be spatially random)",
                               fontweight="bold")
ax_moran_bands_map.set_xlabel("East  →")
ax_moran_bands_map.set_ylabel("North  →")
plt.colorbar(im_resid, ax=ax_moran_bands_map, label="log(O/E)")

# ── Plot 7: Thin territory error comparison ────────────────────────────────
n_thin_show = min(30, n_thin)
thin_show_idx = thin_idx[np.argsort(exposure[thin_idx])][:n_thin_show]
x_thin = np.arange(n_thin_show)
width = 0.26

err_grand_thin = np.abs(pred_grand[thin_show_idx] - true_rate[thin_show_idx])
err_bands_thin = np.abs(pred_bands[thin_show_idx] - true_rate[thin_show_idx])
err_bym2_thin  = np.abs(pred_bym2[thin_show_idx]  - true_rate[thin_show_idx])

ax_thin.bar(x_thin - width, err_grand_thin, width, label="Grand mean", color=GRAND_COLOUR, alpha=0.80)
ax_thin.bar(x_thin,         err_bands_thin, width, label="Bands",      color=BANDS_COLOUR, alpha=0.80)
ax_thin.bar(x_thin + width, err_bym2_thin,  width, label="BYM2",       color=BYM2_COLOUR,  alpha=0.85)

# Annotate with exposure
for i, idx in enumerate(thin_show_idx):
    ax_thin.text(i, -0.002, f"n={int(exposure[idx])}", ha="center", va="top", fontsize=6, rotation=90)

ax_thin.set_xlabel("Territory (sorted by exposure, thinnest first)  /  n = policies")
ax_thin.set_ylabel("|Estimated − True| rate")
ax_thin.set_title(f"Absolute Error on {n_thin_show} Thinnest Territories\n"
                   "(sorted left-to-right by increasing exposure)",
                   fontweight="bold")
ax_thin.legend(fontsize=9)
ax_thin.grid(True, alpha=0.25, axis="y")
ax_thin.set_ylim(bottom=-0.012)

plt.suptitle(
    "insurance-spatial: BYM2 spatial smoothing vs territory banding vs grand mean\n"
    f"Synthetic DGP — {N_AREAS} territories, known true rates, spatially correlated Poisson claims",
    fontsize=13, fontweight="bold"
)
plt.savefig("/tmp/benchmark_spatial_main.png", dpi=120, bbox_inches="tight")
plt.show()
print("Main benchmark plot saved to /tmp/benchmark_spatial_main.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Spatial Shrinkage: What BYM2 Actually Does to Thin Territories

# COMMAND ----------

# MAGIC %md
# MAGIC The clearest way to understand BYM2 is to look at the shrinkage it applies.
# MAGIC For each territory, the observed frequency is a noisy measurement; BYM2 returns an
# MAGIC estimate that has been shrunk toward a spatially weighted neighbourhood average.
# MAGIC The degree of shrinkage depends on:
# MAGIC
# MAGIC - Exposure: thin territories shrink more (their Poisson likelihood is weak)
# MAGIC - Neighbourhood consistency: territories where all neighbours agree shrink less
# MAGIC   (the local spatial signal is clear); territories in heterogeneous patches shrink more
# MAGIC
# MAGIC We quantify this as the "shrinkage fraction":
# MAGIC
# MAGIC     shrinkage = 1 - (|bym2 - obs_freq|) / (|grand_mean - obs_freq|)
# MAGIC
# MAGIC A value near 1 means BYM2 ignored the observed rate entirely and returned the portfolio
# MAGIC mean (maximum shrinkage — appropriate for territories with zero or near-zero exposure).
# MAGIC A value near 0 means BYM2 trusted the observed rate (appropriate for thick territories).
# MAGIC Negative values mean BYM2 moved the estimate in the opposite direction to the grand mean
# MAGIC shrinkage — this happens when spatial neighbours agree on a rate that differs from the
# MAGIC portfolio mean.

# COMMAND ----------

# Shrinkage toward grand mean
denom_shrink = np.abs(pred_grand - obs_freq)
# Avoid division by zero for territories where obs_freq == grand_mean exactly
denom_shrink_safe = np.where(denom_shrink < 1e-8, 1e-8, denom_shrink)
shrinkage_toward_grand = 1 - np.abs(pred_bym2 - obs_freq) / denom_shrink_safe

# Also quantify shrinkage in absolute terms: how much did BYM2 move the estimate?
bym2_move = pred_bym2 - obs_freq  # positive = moved up toward grand mean

print("Shrinkage analysis: how much does BYM2 move estimates away from raw observation?")
print()
print("Thin territories (shrinkage toward grand mean):")
print(f"  Mean:   {shrinkage_toward_grand[is_thin].mean():.3f}")
print(f"  Median: {np.median(shrinkage_toward_grand[is_thin]):.3f}")
print(f"  Min:    {shrinkage_toward_grand[is_thin].min():.3f}")
print(f"  Max:    {shrinkage_toward_grand[is_thin].max():.3f}")
print()
print("Thick territories (shrinkage toward grand mean):")
print(f"  Mean:   {shrinkage_toward_grand[~is_thin].mean():.3f}")
print(f"  Median: {np.median(shrinkage_toward_grand[~is_thin]):.3f}")
print(f"  Min:    {shrinkage_toward_grand[~is_thin].min():.3f}")
print(f"  Max:    {shrinkage_toward_grand[~is_thin].max():.3f}")

print()
print("Absolute BYM2 adjustment (pred_bym2 - obs_freq):")
print(f"  Thin  — Mean abs adjustment: {np.abs(bym2_move[is_thin]).mean():.5f}")
print(f"  Thick — Mean abs adjustment: {np.abs(bym2_move[~is_thin]).mean():.5f}")

# COMMAND ----------

# Scatter: shrinkage vs log(exposure)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax_sh = axes[0]
ax_sh.scatter(np.log(exposure[is_thin]),  shrinkage_toward_grand[is_thin],
               color=THIN_COLOUR, alpha=0.7, s=40, label=f"Thin (n={n_thin})")
ax_sh.scatter(np.log(exposure[~is_thin]), shrinkage_toward_grand[~is_thin],
               color=THICK_COLOUR, alpha=0.5, s=30, label=f"Thick (n={n_thick})")
ax_sh.axhline(1.0, color="grey", lw=1, linestyle="--", alpha=0.7, label="Full shrinkage to grand mean")
ax_sh.axhline(0.0, color="grey", lw=1, linestyle=":",  alpha=0.5, label="No shrinkage (raw = BYM2)")
ax_sh.set_xlabel("log(exposure)")
ax_sh.set_ylabel("Shrinkage fraction toward grand mean")
ax_sh.set_title("BYM2 Shrinkage vs Exposure\n(thin territories shrink more)", fontweight="bold")
ax_sh.legend(fontsize=8)
ax_sh.grid(True, alpha=0.3)

# Uncertainty (posterior SD) vs exposure
ax_unc = axes[1]
ax_unc.scatter(np.log(exposure[is_thin]),  mu_sd_per_area[is_thin],
                color=THIN_COLOUR, alpha=0.7, s=40, label=f"Thin (n={n_thin})")
ax_unc.scatter(np.log(exposure[~is_thin]), mu_sd_per_area[~is_thin],
                color=THICK_COLOUR, alpha=0.5, s=30, label=f"Thick (n={n_thick})")
ax_unc.set_xlabel("log(exposure)")
ax_unc.set_ylabel("Posterior SD of estimated rate")
ax_unc.set_title("BYM2 Uncertainty vs Exposure\n(thin territories have wider posterior)",
                  fontweight="bold")
ax_unc.legend(fontsize=8)
ax_unc.grid(True, alpha=0.3)

plt.suptitle("BYM2 shrinkage and uncertainty quantification per territory",
              fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/benchmark_spatial_shrinkage.png", dpi=120, bbox_inches="tight")
plt.show()
print("Shrinkage plot saved to /tmp/benchmark_spatial_shrinkage.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Territory Relativities: Production Output

# COMMAND ----------

# MAGIC %md
# MAGIC In production, you do not use the raw estimated rate directly. You use the territory
# MAGIC *relativity* — a multiplicative factor expressing how much this territory's expected
# MAGIC frequency differs from the base territory or the portfolio mean.
# MAGIC
# MAGIC The relativity table from `territory_relativities()` is ready to load directly into a
# MAGIC GLM as a log-scale offset column (the `ln_offset` column). This is the standard way to
# MAGIC incorporate a territory model into an existing multiplicative pricing framework without
# MAGIC re-fitting the whole GLM.
# MAGIC
# MAGIC The credibility interval columns tell you which relativities you can trust. A territory
# MAGIC with relativity 1.15 and 95% credibility interval [1.10, 1.21] is unambiguously high-risk.
# MAGIC A territory with relativity 1.15 and interval [0.85, 1.52] is statistically indistinguishable
# MAGIC from average — you should be cautious about applying a loading.

# COMMAND ----------

rels = bym2_result.territory_relativities(credibility_interval=0.95)

print("Territory relativities (first 20 rows, sorted by relativity):")
print(
    rels.sort("relativity", descending=True)
    .head(20)
    .with_columns([
        pl.col("relativity").round(4),
        pl.col("lower").round(4),
        pl.col("upper").round(4),
        pl.col("b_mean").round(4),
        pl.col("b_sd").round(4),
        pl.col("ln_offset").round(4),
    ])
    .to_pandas()
    .to_string(index=False)
)

print(f"\nRelativity distribution:")
rel_arr = rels["relativity"].to_numpy()
print(f"  Min:    {rel_arr.min():.4f}")
print(f"  Median: {np.median(rel_arr):.4f}")
print(f"  Mean:   {rel_arr.mean():.4f}")
print(f"  Max:    {rel_arr.max():.4f}")
print(f"  SD:     {rel_arr.std():.4f}")

# COMMAND ----------

# Width of the 95% credibility interval as a function of exposure
ci_width = rels["upper"].to_numpy() - rels["lower"].to_numpy()

print("Credibility interval width by exposure band:")
exp_bands = [
    ("Very thin (< 20)",  exposure < 20),
    ("Thin (20–50)",      (exposure >= 20) & (exposure < 50)),
    ("Moderate (50–200)", (exposure >= 50) & (exposure < 200)),
    ("Thick (200+)",      exposure >= 200),
]
for label, mask in exp_bands:
    n_in_band = int(mask.sum())
    if n_in_band == 0:
        continue
    print(f"  {label}: n={n_in_band:3d}  mean CI width = {ci_width[mask].mean():.4f}  "
          f"(relativity range [{rels['lower'].to_numpy()[mask].mean():.3f}, "
          f"{rels['upper'].to_numpy()[mask].mean():.3f}])")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Runtime Summary

# COMMAND ----------

print("Runtime summary")
print("=" * 50)
print(f"  Data generation:          {0.001:.3f}s (negligible)")
print(f"  Moran's I (pre-fit):      {moran_raw_time:.2f}s")
print(f"  Grand mean baseline:      <1ms")
print(f"  Territory banding:        {baseline_bands_time*1000:.1f}ms")
print(f"  BYM2 MCMC fit:            {bym2_fit_time:.1f}s  ({bym2_fit_time/60:.1f} min)")
print(f"  Moran's I (post-fit, ×3): {moran_post_time:.2f}s")
print()
print("BYM2 model configuration:")
print(f"  Draws:   {bym2.draws} per chain  ×  {bym2.chains} chains = {bym2.draws * bym2.chains} total samples")
print(f"  Tune:    {bym2.tune} steps per chain")
print(f"  Areas:   {adj.n}")
print()
print("The MCMC runtime is the tradeoff. On a UK portfolio with ~2,800 postcode districts,")
print("a production BYM2 run with 1000 draws and 4 chains takes roughly 15–30 minutes.")
print("Run it quarterly. Cache the relativity table. The marginal cost per policy is zero.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Verdict

# COMMAND ----------

# MAGIC %md
# MAGIC ### When BYM2 spatial smoothing earns its keep
# MAGIC
# MAGIC **BYM2 wins over territory banding when:**
# MAGIC - The portfolio has genuine spatial autocorrelation in claim frequency — neighbouring
# MAGIC   postcode sectors have more similar risk than distant ones. This is true for nearly
# MAGIC   every UK personal lines line of business driven by environmental risk (motor theft,
# MAGIC   subsidence, flood, escape of water from ageing pipe infrastructure).
# MAGIC - A substantial fraction of territories are thin. In UK personal lines, postcode-sector
# MAGIC   level typically gives you ~10,000 territories against a book of 200,000–500,000 policies.
# MAGIC   Most sectors have tens of policies, not thousands.
# MAGIC - You need uncertainty quantification on territory rates. The posterior credibility
# MAGIC   intervals tell you which relativities to trust and which to treat as noise.
# MAGIC
# MAGIC **BYM2 wins over the grand mean when:**
# MAGIC - There is genuine geographic variation in risk (i.e., always, for territorial lines).
# MAGIC - You can construct a valid adjacency graph. This requires contiguous territories —
# MAGIC   standard for UK postcode sector level, but problematic for categorical groupings
# MAGIC   like occupation class.
# MAGIC
# MAGIC **BYM2 is not the right tool when:**
# MAGIC - Territories are not spatially contiguous (e.g., your territory variable is a
# MAGIC   "region code" that does not map to a physical adjacency structure).
# MAGIC - The graph has many disconnected components that cannot be connected meaningfully.
# MAGIC - MCMC runtime is prohibitive and you cannot run it offline. This is a workflow
# MAGIC   constraint, not a statistical one — the answer is to cache the territory relativities
# MAGIC   and update them quarterly, not to abandon the model.
# MAGIC - You have fewer than ~30 territories. With very few areas, the spatial precision
# MAGIC   parameters are poorly identified and the model degrades to a poorly-calibrated
# MAGIC   version of the grand mean.
# MAGIC
# MAGIC **What Moran's I tells you:**
# MAGIC
# MAGIC - Before fitting: significant positive I confirms that spatial smoothing is warranted.
# MAGIC   If I is not significant pre-fit, the spatial model will not add much over a
# MAGIC   non-spatial random effects model.
# MAGIC - After fitting BYM2: near-zero I on residuals confirms the model has absorbed the
# MAGIC   spatial structure. Residual autocorrelation means the model missed something —
# MAGIC   possibly a non-contiguous spatial effect or a missing covariate.
# MAGIC - After banding: still-significant I on residuals means the band edges are misaligned
# MAGIC   with the true spatial structure. This is the expected result for quintile banding
# MAGIC   based on noisy raw frequencies.
# MAGIC
# MAGIC **The MSE story in this benchmark:**
# MAGIC
# MAGIC The key number is not the ratio between methods — that depends on the specific DGP
# MAGIC parameters (how much spatial correlation, how thin the thinnest territories, how smooth
# MAGIC the underlying surface). The key insight is the mechanism: BYM2 allocates its smoothing
# MAGIC adaptively across the graph, with the ICAR prior doing the work of identifying which
# MAGIC territories are well-supported by their neighbours and which are isolated. Quintile banding
# MAGIC has no such mechanism — it treats a thin rural territory with 8 policies the same way it
# MAGIC treats a thick urban territory with 600 policies, as long as their raw frequencies are
# MAGIC in the same quintile.

# COMMAND ----------

# Print the numeric verdict
print("=" * 80)
print("VERDICT: BYM2 vs territory banding vs grand mean")
print("=" * 80)
print()
print(metrics_df.to_string(index=False))
print()
print("Moran's I on residuals (spatial autocorrelation remaining after fitting):")
for label, m in [("Grand mean", moran_grand), ("Bands", moran_bands), ("BYM2", moran_bym2)]:
    sig_str = "SIGNIFICANT" if m.significant else "not significant"
    print(f"  {label:18s}  I = {m.statistic:+.4f}   p = {m.p_value:.4f}   ({sig_str})")

print()
print("Band assignment accuracy for thin territories vs ground truth:")
print(f"  {thin_band_correct.mean()*100:.0f}% of thin territories placed in the correct band")
print(f"  ({(band_jump >= 2).mean()*100:.0f}% were off by 2+ bands)")
print()
print(f"BYM2 rho (proportion of spatial variance that is structured): {rho_mean:.3f}")
print(f"BYM2 sigma (total SD of territory log-rate effects):          {sigma_mean:.3f}")
print(f"MCMC fit time: {bym2_fit_time:.0f}s ({bym2_fit_time/60:.1f} min)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. README Performance Snippet

# COMMAND ----------

# Auto-generate the Performance section text for the library README.
# These numbers come directly from the benchmark run above.

rmse_grand_all  = np.sqrt(mse_grand_all)
rmse_bands_all  = np.sqrt(mse_bands_all)
rmse_bym2_all   = np.sqrt(mse_bym2_all)

rmse_grand_thin = np.sqrt(mse_grand_thin)
rmse_bands_thin = np.sqrt(mse_bands_thin)
rmse_bym2_thin  = np.sqrt(mse_bym2_thin)

rmse_grand_thick = np.sqrt(mse_grand_thick)
rmse_bands_thick = np.sqrt(mse_bands_thick)
rmse_bym2_thick  = np.sqrt(mse_bym2_thick)

readme_snippet = f"""
## Performance

Benchmarked on synthetic Poisson frequency data with known ground truth — {N_AREAS} territories
on a {NROWS}×{NCOLS} grid with genuine spatial autocorrelation, {n_thin} thin territories
(exposure < {THIN_THRESHOLD} policies), and {n_thick} thick territories. True rates known from the DGP.
See `notebooks/benchmark.py` for full methodology.

**MSE vs true DGP territory rates:**

| Subset               | Grand mean RMSE | Bands RMSE | BYM2 RMSE | BYM2 vs Bands |
|----------------------|-----------------|------------|-----------|---------------|
| All (n={N_AREAS})       | {rmse_grand_all:.5f}     | {rmse_bands_all:.5f}   | {rmse_bym2_all:.5f} | {pct_vs(mse_bands_all, mse_bym2_all):+.1f}% MSE    |
| Thin (n={n_thin})       | {rmse_grand_thin:.5f}     | {rmse_bands_thin:.5f}   | {rmse_bym2_thin:.5f} | {pct_vs(mse_bands_thin, mse_bym2_thin):+.1f}% MSE    |
| Thick (n={n_thick})      | {rmse_grand_thick:.5f}     | {rmse_bands_thick:.5f}   | {rmse_bym2_thick:.5f} | {pct_vs(mse_bands_thick, mse_bym2_thick):+.1f}% MSE    |

**Moran's I on residuals** (spatial autocorrelation remaining after fitting):

| Model      | Moran's I | p-value | Significant? |
|------------|-----------|---------|--------------|
| Grand mean | {moran_grand.statistic:.4f}    | {moran_grand.p_value:.4f}  | {"Yes" if moran_grand.significant else "No"} |
| Bands      | {moran_bands.statistic:.4f}    | {moran_bands.p_value:.4f}  | {"Yes" if moran_bands.significant else "No"} |
| BYM2       | {moran_bym2.statistic:.4f}    | {moran_bym2.p_value:.4f}  | {"Yes" if moran_bym2.significant else "No"} |

Posterior rho = {rho_mean:.3f} (proportion of geographic variance that is spatially structured).
"""

print(readme_snippet)
