# Databricks notebook source
# MAGIC %md
# MAGIC # BYM2 INLA Territory Model - Demo
# MAGIC
# MAGIC This notebook demonstrates `BYM2InlaModel` — the pyINLA backend for the
# MAGIC `insurance-spatial` library.  INLA (Integrated Nested Laplace Approximation)
# MAGIC is orders of magnitude faster than MCMC for models in the Latent Gaussian
# MAGIC Model class, and BYM2 is squarely in that class.
# MAGIC
# MAGIC **Benchmark context:** pyINLA (KAUST, arXiv:2603.27276, March 2026) is 278x
# MAGIC faster than PyMC NUTS on the Scottish lip cancer dataset (N=56).  For UK
# MAGIC postcode sector models with N=8,000 areas, MCMC takes hours; INLA takes
# MAGIC seconds.
# MAGIC
# MAGIC **What this notebook covers:**
# MAGIC 1. Install dependencies
# MAGIC 2. Build a 10×10 synthetic territory grid (N=100)
# MAGIC 3. Generate synthetic claim data with a known north-south spatial gradient
# MAGIC 4. Fit BYM2InlaModel — the INLA backend
# MAGIC 5. Fit BYM2Model — the MCMC backend (for comparison)
# MAGIC 6. Compare relativities from both backends
# MAGIC 7. Benchmark timing: INLA vs MCMC
# MAGIC 8. Review INLA-specific diagnostics (DIC, WAIC, hyperparameter posteriors)
# MAGIC
# MAGIC **Platform note:** pyINLA wheels are available for Linux x86_64, macOS ARM64,
# MAGIC and macOS x86_64.  Databricks runs on Linux x86_64 — fully supported.

# COMMAND ----------

# MAGIC %pip install 'insurance-spatial[inla,mcmc]' pymc arviz polars

# COMMAND ----------

# MAGIC %md ## 1. Build a 10×10 grid adjacency matrix (N=100)

# COMMAND ----------

import time
import numpy as np
import polars as pl

from insurance_spatial.adjacency import build_grid_adjacency
from insurance_spatial.diagnostics import moran_i

# 10×10 rook-contiguity grid: 100 territory areas
adj = build_grid_adjacency(10, 10, connectivity="rook")
print(f"Areas: {adj.n}")
print(f"Connected components: {adj.n_components()}")
print(f"BYM2 scaling factor: {adj.scaling_factor:.4f}")
print(f"Sample areas: {adj.areas[:5]}")

# COMMAND ----------

# MAGIC %md ## 2. Simulate claim data with a spatial gradient
# MAGIC
# MAGIC We embed a north-south gradient (rows 0-4 higher risk, rows 5-9 lower risk)
# MAGIC plus some IID noise.  This gives Moran's I ≈ 0.4, a level of spatial
# MAGIC autocorrelation typical of UK motor frequency data.

# COMMAND ----------

rng = np.random.default_rng(2026)
N = adj.n  # 100

# Exposure: policy years per area (varies between 50 and 800)
exposure = rng.uniform(50.0, 800.0, size=N)

# True spatial log-rate: north (rows 0-4) is 40% higher risk than south (rows 5-9)
# Plus IID noise with SD=0.15
row_idx = np.array([i // 10 for i in range(N)])  # 0..9
true_log_rate = 0.4 - 0.08 * row_idx + 0.15 * rng.standard_normal(N)

# Observed claims from Poisson
mu = exposure * np.exp(true_log_rate)
claims = rng.poisson(mu)

print(f"Total claims: {claims.sum()}")
print(f"Mean frequency: {(claims / exposure).mean():.4f}")
print(f"Min/max frequency: {(claims/exposure).min():.4f} / {(claims/exposure).max():.4f}")

# COMMAND ----------

# MAGIC %md ## 3. Check for spatial autocorrelation (Moran's I)
# MAGIC
# MAGIC Always do this before fitting a spatial model.  A non-significant Moran's I
# MAGIC means spatial smoothing is not warranted and BYM2 will just waste compute.

# COMMAND ----------

log_oe = np.log((claims / exposure) / (claims / exposure).mean())
moran_result = moran_i(log_oe, adj, n_permutations=999)

print(f"Moran's I: {moran_result.statistic:.4f}")
print(f"Expected:  {moran_result.expected:.4f}")
print(f"p-value:   {moran_result.p_value:.4f}")
print(f"Significant: {moran_result.significant}")
print(f"\n{moran_result.interpretation}")

# COMMAND ----------

# MAGIC %md ## 4. Fit BYM2InlaModel (INLA backend)
# MAGIC
# MAGIC This is the fast path.  On N=100 we expect well under 1 second.

# COMMAND ----------

from insurance_spatial.bym2_inla import BYM2InlaModel

model_inla = BYM2InlaModel(
    adjacency=adj,
    # PC prior: P(sigma > 1) = 0.01 — weak prior, data dominates
    sigma_u_prior=(1.0, 0.01),
    # PC prior: P(rho > 0.5) = 0.5 — symmetric mixing prior
    rho_prior=(0.5, 0.5),
)

t0 = time.perf_counter()
result_inla = model_inla.fit(claims=claims, exposure=exposure)
elapsed_inla = time.perf_counter() - t0

print(f"INLA fit time: {elapsed_inla:.3f}s")
print(f"Areas: {result_inla.n_areas}")
print(f"DIC: {result_inla.dic:.2f}")
print(f"WAIC: {result_inla.waic:.2f}")

# COMMAND ----------

# MAGIC %md ### Hyperparameter posteriors
# MAGIC
# MAGIC sigma is the total standard deviation of the spatial random effect on the
# MAGIC log scale.  rho is the proportion of that variance attributable to genuine
# MAGIC spatial structure (vs. IID noise).  rho near 1.0 means the residual
# MAGIC geographic variation is spatially smooth.

# COMMAND ----------

print("Hyperparameter posteriors:")
print(result_inla.hyperpar_summary)

# COMMAND ----------

# MAGIC %md ### Territory relativities

# COMMAND ----------

rels_inla = result_inla.territory_relativities()
print(f"Schema: {rels_inla.schema}")
print(f"\nTop 10 highest-risk areas:")
print(
    rels_inla
    .sort("relativity", descending=True)
    .head(10)
    .select(["area", "relativity", "lower", "upper", "ln_offset"])
)

# COMMAND ----------

# MAGIC %md ## 5. Fit BYM2Model (MCMC baseline)
# MAGIC
# MAGIC We use a short chain (draws=200, chains=2) to keep runtime manageable.
# MAGIC In production you would use draws=1000, chains=4 and verify convergence.

# COMMAND ----------

from insurance_spatial.models import BYM2Model

model_mcmc = BYM2Model(
    adjacency=adj,
    draws=200,
    chains=2,
    tune=300,
    target_accept=0.9,
)

t0 = time.perf_counter()
result_mcmc = model_mcmc.fit(claims=claims, exposure=exposure, random_seed=42)
elapsed_mcmc = time.perf_counter() - t0

print(f"MCMC fit time: {elapsed_mcmc:.1f}s")

# COMMAND ----------

# MAGIC %md ## 6. Compare INLA vs MCMC relativities
# MAGIC
# MAGIC If the two backends agree closely (r > 0.95), we have confidence that the
# MAGIC INLA approximation is accurate for this dataset.  For BYM2 with N > 50,
# MAGIC the Gaussian approximation used by INLA is generally excellent.

# COMMAND ----------

rels_mcmc = result_mcmc.territory_relativities()

# Join on area
comparison = (
    rels_inla
    .rename({"relativity": "rel_inla", "b_mean": "b_mean_inla"})
    .join(
        rels_mcmc
        .rename({"relativity": "rel_mcmc", "b_mean": "b_mean_mcmc"}),
        on="area",
    )
    .select(["area", "b_mean_inla", "b_mean_mcmc", "rel_inla", "rel_mcmc"])
)

# Pearson correlation on log-scale b_mean
b_inla = comparison["b_mean_inla"].to_numpy()
b_mcmc = comparison["b_mean_mcmc"].to_numpy()
correlation = float(np.corrcoef(b_inla, b_mcmc)[0, 1])

print(f"Pearson correlation of b_mean (INLA vs MCMC): {correlation:.4f}")
print(f"\nExpected: r > 0.95 for a well-identified spatial model")
print(f"\nFirst 10 rows:")
print(comparison.head(10))

# COMMAND ----------

# MAGIC %md ## 7. Timing summary

# COMMAND ----------

speedup = elapsed_mcmc / elapsed_inla if elapsed_inla > 0 else float("inf")
print(f"INLA: {elapsed_inla:.3f}s")
print(f"MCMC: {elapsed_mcmc:.1f}s")
print(f"Speedup: {speedup:.0f}x")
print()
print("Published benchmark (pyINLA paper, arXiv:2603.27276):")
print("  N=56 (Scottish lip cancer): INLA 0.42s vs PyMC NUTS 117.2s = 278x speedup")

# COMMAND ----------

# MAGIC %md ## 8. INLA-specific diagnostics

# COMMAND ----------

diag = result_inla.diagnostics()

print("INLA Diagnostics")
print("================")
print(f"DIC:  {diag.dic:.2f}  (lower = better fit, penalises complexity)")
print(f"WAIC: {diag.waic:.2f}  (lower = better fit, more robust than DIC)")
print()
print("Hyperparameter posteriors:")
print(diag.hyperpar_summary)
print()
print("Fixed effects posteriors:")
print(diag.fixed_summary)

# COMMAND ----------

# MAGIC %md ## 9. Two-stage workflow: INLA on O/E residuals
# MAGIC
# MAGIC In production the recommended approach is:
# MAGIC 1. Fit a non-spatial GLM (or LightGBM) to capture all the structured rating
# MAGIC    factors (vehicle group, driver age, NCD, etc.)
# MAGIC 2. Compute sector-level O/E ratios: observed claims / expected claims
# MAGIC 3. Pass the O/E ratios into BYM2InlaModel to capture the residual geographic
# MAGIC    structure
# MAGIC
# MAGIC This keeps the spatial model interpretable and decoupled from the base model.
# MAGIC Here we simulate step 3 by using our synthetic data as if it were O/E ratios.

# COMMAND ----------

# Simulate: observed = claims, exposure = expected from base model (just use raw here)
oe_claims = claims  # in reality: actual claims per sector
oe_exposure = exposure  # in reality: GLM-predicted expected claims per sector

model_two_stage = BYM2InlaModel(adjacency=adj)
result_two_stage = model_two_stage.fit(claims=oe_claims, exposure=oe_exposure)
rels_two_stage = result_two_stage.territory_relativities()

print("Two-stage territorial relativities (sample):")
print(
    rels_two_stage
    .sort("relativity", descending=True)
    .head(5)
    .select(["area", "relativity", "lower", "upper"])
)

# COMMAND ----------

# MAGIC %md ## 10. Writing relativities back to a Spark DataFrame
# MAGIC
# MAGIC The ln_offset column is ready to use as a log-scale offset in a downstream
# MAGIC Spark GLM or as a lookup table keyed by postcode sector.

# COMMAND ----------

# Convert to Spark DataFrame for storage or joining to policy-level data
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Convert Polars to pandas first, then to Spark
rels_spark = spark.createDataFrame(rels_inla.to_pandas())
rels_spark.createOrReplaceTempView("territory_relativities_inla")

print(f"Written {rels_spark.count()} rows to Spark view: territory_relativities_inla")
print("Schema:")
rels_spark.printSchema()

# COMMAND ----------

# MAGIC %md ## Summary
# MAGIC
# MAGIC | Backend | Fit time (N=100) | Mechanism | Intervals |
# MAGIC |---------|-----------------|-----------|-----------|
# MAGIC | BYM2InlaModel | < 1s | INLA Laplace approximation | Gaussian marginal |
# MAGIC | BYM2Model | ~minutes | MCMC NUTS | Exact posterior |
# MAGIC
# MAGIC **When to use which:**
# MAGIC - INLA for large N (> 500) and iterative production reruns
# MAGIC - MCMC for prior sensitivity, convergence validation, and smaller N
# MAGIC - Both backends produce the same `territory_relativities()` schema
# MAGIC
# MAGIC **Install:**
# MAGIC ```
# MAGIC # INLA backend:
# MAGIC uv add 'insurance-spatial[inla]'
# MAGIC
# MAGIC # MCMC backend:
# MAGIC uv add 'insurance-spatial[mcmc]'
# MAGIC
# MAGIC # Both:
# MAGIC uv add 'insurance-spatial[inla,mcmc]'
# MAGIC ```
