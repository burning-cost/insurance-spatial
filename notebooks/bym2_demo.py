# Databricks notebook source
# MAGIC %md
# MAGIC # BYM2 Spatial Territory Ratemaking — Demo
# MAGIC
# MAGIC This notebook demonstrates the full `insurance-spatial` workflow on synthetic UK motor data.
# MAGIC It is designed to run on Databricks serverless compute (Python 3.10+).
# MAGIC
# MAGIC **What this notebook covers:**
# MAGIC 1. Build an adjacency matrix for a synthetic 10×10 territory grid
# MAGIC 2. Generate synthetic claim data with genuine spatial autocorrelation
# MAGIC 3. Test for spatial autocorrelation (Moran's I) on raw O/E residuals
# MAGIC 4. Fit a BYM2 Poisson territory model via PyMC
# MAGIC 5. Inspect convergence diagnostics (R-hat, ESS, divergences)
# MAGIC 6. Extract territory relativities as a Polars DataFrame
# MAGIC 7. Interpret the rho parameter — how much of the variation is genuinely spatial?

# COMMAND ----------

# MAGIC %pip install insurance-spatial pymc arviz polars

# COMMAND ----------

# MAGIC %md ## 1. Setup

# COMMAND ----------

import numpy as np
import polars as pl

from insurance_spatial.adjacency import build_grid_adjacency
from insurance_spatial.models import BYM2Model
from insurance_spatial.diagnostics import moran_i, convergence_summary
from insurance_spatial.relativities import extract_relativities

# COMMAND ----------

# MAGIC %md ## 2. Build Adjacency Matrix
# MAGIC
# MAGIC We simulate a 10×10 grid of postcode sectors with Queen contiguity.
# MAGIC In production, you would load this from a GeoJSON of actual sector boundaries:
# MAGIC
# MAGIC ```python
# MAGIC from insurance_spatial.adjacency import from_geojson
# MAGIC adj = from_geojson("postcode_sectors.geojson", area_col="PC_SECTOR")
# MAGIC ```

# COMMAND ----------

NROWS, NCOLS = 10, 10
adj = build_grid_adjacency(NROWS, NCOLS, connectivity="queen")

print(f"Number of areas: {adj.n}")
print(f"Connected components: {adj.n_components()} (must be 1 for ICAR)")
print(f"BYM2 scaling factor: {adj.scaling_factor:.4f}")
print(f"Mean neighbours per area: {adj.neighbour_counts().mean():.1f}")
print(f"\nFirst 5 area labels: {adj.areas[:5]}")

# COMMAND ----------

# MAGIC %md ## 3. Generate Synthetic Spatial Claims Data
# MAGIC
# MAGIC We create a territory risk surface with:
# MAGIC - A north–south gradient (top rows have higher theft risk)
# MAGIC - A "hot spot" cluster around the centre-right (urban area)
# MAGIC - Noise
# MAGIC
# MAGIC This mimics the kind of spatial pattern you see in UK motor theft:
# MAGIC higher risk in inner-city areas, lower in rural periphery.

# COMMAND ----------

rng = np.random.default_rng(2024)
N = adj.n

# Spatial coordinates (row and column indices)
row_idx = np.array([i // NCOLS for i in range(N)])
col_idx = np.array([i % NCOLS for i in range(N)])

# True log-rate surface
# North-south gradient
north_south = 0.4 * (1.0 - row_idx / (NROWS - 1))
# Urban hot-spot: Gaussian bump centred at (row=6, col=7)
hot_spot = 0.6 * np.exp(
    -0.5 * ((row_idx - 6) ** 2 + (col_idx - 7) ** 2) / 4.0
)
# Unstructured noise
noise = 0.15 * rng.standard_normal(N)

true_log_rate = north_south + hot_spot + noise
# Centre so overall rate is moderate
true_log_rate -= true_log_rate.mean()

# Exposure: urban areas (centre) have more policies
base_exposure = 200.0
exposure_factor = 1.0 + 3.0 * np.exp(
    -0.5 * ((row_idx - 5) ** 2 + (col_idx - 5) ** 2) / 8.0
)
exposure = base_exposure * exposure_factor

# Observed claims
mu = exposure * np.exp(true_log_rate)
claims = rng.poisson(mu).astype(np.int64)

print(f"Total claims: {claims.sum():,}")
print(f"Total exposure: {exposure.sum():,.0f} policy-years")
print(f"Mean claim rate: {claims.sum() / exposure.sum():.4f}")
print(f"\nClaims range: {claims.min()} – {claims.max()}")
print(f"Sectors with zero claims: {(claims == 0).sum()}")

# COMMAND ----------

# MAGIC %md ## 4. Pre-fit Moran's I Test
# MAGIC
# MAGIC We test whether there is significant spatial autocorrelation in the
# MAGIC raw log(O/E) residuals.  A significant positive Moran's I confirms
# MAGIC that spatial smoothing is warranted.

# COMMAND ----------

# O/E residuals (using overall mean as expected)
overall_rate = claims.sum() / exposure.sum()
expected = exposure * overall_rate
log_oe = np.log((claims + 0.5) / (expected + 0.5))  # +0.5 Laplace smoothing

moran_pre = moran_i(log_oe, adj, n_permutations=999)

print("=== Moran's I on log(O/E) residuals (pre-fit) ===")
print(f"  Moran's I:  {moran_pre.statistic:.4f}")
print(f"  Expected:   {moran_pre.expected:.4f}")
print(f"  Z-score:    {moran_pre.z_score:.2f}")
print(f"  p-value:    {moran_pre.p_value:.4f}")
print(f"  Significant: {moran_pre.significant}")
print(f"\n  {moran_pre.interpretation}")

# COMMAND ----------

# MAGIC %md ## 5. Fit the BYM2 Model
# MAGIC
# MAGIC We use short chains (500 draws × 4 chains) for the demo.
# MAGIC For production use on real data, increase to 1,000–2,000 draws.
# MAGIC
# MAGIC The model:
# MAGIC
# MAGIC ```
# MAGIC y_i ~ Poisson(mu_i)
# MAGIC log(mu_i) = log(E_i) + alpha + b_i
# MAGIC b_i = sigma * (sqrt(rho/s) * phi_i + sqrt(1-rho) * theta_i)
# MAGIC phi ~ ICAR(W)          # structured spatial
# MAGIC theta ~ Normal(0, 1)   # unstructured IID
# MAGIC sigma ~ HalfNormal(1)
# MAGIC rho ~ Beta(0.5, 0.5)
# MAGIC ```

# COMMAND ----------

model = BYM2Model(
    adjacency=adj,
    draws=500,
    chains=4,
    tune=800,
    target_accept=0.9,
)

result = model.fit(
    claims=claims,
    exposure=exposure,
    random_seed=42,
)

print("Model fitting complete.")

# COMMAND ----------

# MAGIC %md ## 6. Convergence Diagnostics

# COMMAND ----------

conv = convergence_summary(result)

print("=== Convergence Summary ===")
print(f"  Max R-hat:     {conv.max_rhat:.4f}  (target: < 1.01)")
print(f"  Min ESS bulk:  {conv.min_ess_bulk:.0f}  (target: > 400)")
print(f"  Min ESS tail:  {conv.min_ess_tail:.0f}  (target: > 400)")
print(f"  Divergences:   {conv.n_divergences}")
print(f"  Converged:     {conv.converged}")

print("\n=== R-hat by parameter ===")
print(conv.rhat_by_param)

# COMMAND ----------

# MAGIC %md ## 7. Interpret the Spatial Hyperparameters
# MAGIC
# MAGIC **rho** (spatial proportion): How much of the residual geographic variation
# MAGIC is spatially structured (smooth) vs. pure area-level noise?
# MAGIC - rho → 1: smooth spatial pattern, nearby areas are similar
# MAGIC - rho → 0: no spatial structure, variation is idiosyncratic
# MAGIC
# MAGIC **sigma** (total SD): Overall magnitude of the territory effect.

# COMMAND ----------

diag = result.diagnostics()

print("=== Spatial Hyperparameters ===")
print("\n  rho (spatial proportion):")
print(diag.rho_summary)
print("\n  sigma (total SD):")
print(diag.sigma_summary)

# Quick interpretation
rho_mean = float(diag.rho_summary["mean"][0])
sigma_mean = float(diag.sigma_summary["mean"][0])
print(f"\n  Interpretation:")
print(f"  {100*rho_mean:.0f}% of territory variance is spatially structured.")
if rho_mean > 0.5:
    print("  Strong spatial signal — BYM2 smoothing is adding value.")
else:
    print("  Weak spatial signal — variation is mostly area-specific noise.")
print(f"  Typical territory effect: exp(sigma) ≈ {np.exp(sigma_mean):.2f}x")

# COMMAND ----------

# MAGIC %md ## 8. Extract Territory Relativities

# COMMAND ----------

relativities = result.territory_relativities(credibility_interval=0.95)

print(f"Territory relativities: {len(relativities)} areas")
print("\nTop 5 highest-risk territories:")
print(
    relativities.sort("relativity", descending=True)
    .head(5)
    .select(["area", "relativity", "lower", "upper", "ln_offset"])
)

print("\nTop 5 lowest-risk territories:")
print(
    relativities.sort("relativity")
    .head(5)
    .select(["area", "relativity", "lower", "upper", "ln_offset"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using relativities as a GLM offset
# MAGIC
# MAGIC The `ln_offset` column is log(relativity) — ready to use as a fixed offset
# MAGIC in an Emblem or other GLM:
# MAGIC
# MAGIC ```
# MAGIC GLM: log(mu) = log(E) + log(territory_relativity) + X @ beta
# MAGIC ```
# MAGIC
# MAGIC In Polars, join this to your policy data on the sector code:
# MAGIC
# MAGIC ```python
# MAGIC policies = policies.join(
# MAGIC     relativities.select(["area", "ln_offset"]),
# MAGIC     left_on="postcode_sector",
# MAGIC     right_on="area",
# MAGIC     how="left",
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 9. Post-fit Moran's I
# MAGIC
# MAGIC Check whether the BYM2 model has absorbed the spatial autocorrelation.
# MAGIC The Moran's I on posterior residuals (observed vs. fitted) should be
# MAGIC non-significant if the model has captured the spatial structure.

# COMMAND ----------

# Compute posterior mean fitted values
mu_samples = result.trace.posterior["mu"].values  # (chains, draws, N)
mu_hat = mu_samples.mean(axis=(0, 1))  # (N,)

post_log_oe = np.log((claims + 0.5) / (mu_hat + 0.5))
moran_post = moran_i(post_log_oe, adj, n_permutations=999)

print("=== Moran's I on posterior residuals (post-fit) ===")
print(f"  Moran's I:  {moran_post.statistic:.4f}  (was {moran_pre.statistic:.4f})")
print(f"  p-value:    {moran_post.p_value:.4f}")
print(f"  Significant: {moran_post.significant}")
print(f"\n  {moran_post.interpretation}")

if not moran_post.significant:
    print("\n  ✓ BYM2 has successfully absorbed the spatial autocorrelation.")
else:
    print("\n  ! Residual spatial autocorrelation remains. Consider:")
    print("    - Adding spatial covariates (IMD, crime rate, flood risk)")
    print("    - Checking for islands / disconnected components in the adjacency graph")
    print("    - Increasing draws for better posterior approximation")

# COMMAND ----------

# MAGIC %md ## 10. Summary and Next Steps
# MAGIC
# MAGIC This demo showed the core `insurance-spatial` workflow:
# MAGIC
# MAGIC 1. **Adjacency** — built from a synthetic grid; in production, use `from_geojson()` with ONS sector boundaries
# MAGIC 2. **Pre-fit diagnostic** — Moran's I confirmed spatial autocorrelation, justifying BYM2
# MAGIC 3. **BYM2 fit** — PyMC MCMC with ICAR spatial component and IID noise component
# MAGIC 4. **Convergence** — R-hat and ESS checks (run longer chains for production)
# MAGIC 5. **Relativities** — extracted as multiplicative factors with credibility intervals
# MAGIC 6. **Post-fit diagnostic** — confirmed spatial structure absorbed
# MAGIC
# MAGIC ### Production checklist
# MAGIC
# MAGIC - [ ] Replace synthetic grid with actual ONS postcode sector GeoJSON
# MAGIC - [ ] Run `fix_islands=True` to connect Scottish/Welsh island sectors
# MAGIC - [ ] Increase to `draws=1000, tune=1000, chains=4`
# MAGIC - [ ] Add area-level covariates: IMD score, police.uk vehicle crime rate, EA flood risk band
# MAGIC - [ ] Cache the adjacency matrix and scaling factor (`adj.scaling_factor` is computed once)
# MAGIC - [ ] Save relativities to Delta Lake for downstream GLM consumption

# COMMAND ----------

# Save relativities to Delta (uncomment for production)
# relativities.write_delta("/mnt/pricing/territory/bym2_relativities", mode="overwrite")

print("Demo complete.")
print(f"\nRelativities saved to 'relativities' DataFrame ({len(relativities)} rows)")
print("Columns:", relativities.columns)
