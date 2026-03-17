# insurance-spatial
[![Tests](https://github.com/burning-cost/insurance-spatial/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-spatial/actions/workflows/ci.yml)

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-spatial)

Spatial tools for UK insurance pricing in one install: BYM2 territory ratemaking and spatially weighted conformal prediction intervals.

## The Problems

**Territory pricing** in UK personal lines is broken in predictable ways.

The standard approach — a GLM with postcode sector as a categorical predictor — creates 11,200 separate territory parameters for motor, most of them estimated from a handful of claims. Adjacent sectors can differ by 30–40% on sparse data not because the underlying risk differs but because the estimates are noisy. Standard practice is to band sectors into 6–20 groups using k-means on historical loss ratios. This is ad hoc, discards information, and creates artificial discontinuities at band boundaries.

The academically-grounded alternative is the **BYM2 model** (Besag-York-Mollié, Riebler et al. 2016): a Bayesian hierarchical model that borrows strength across neighbouring postcode sectors, quantifies how much geographic variation is genuinely spatial vs. idiosyncratic noise, and produces territory relativities with proper uncertainty estimates.

**Prediction intervals** from a pricing model are nationally calibrated but geographically broken. Standard conformal prediction gives you a guarantee that 90% of risks are covered nationwide — but the coverage can be 70% in inner London and 98% in rural Somerset. That geographic miscalibration is a conduct risk under Consumer Duty.

The fix is spatially weighted conformal prediction: calibration non-conformity scores are weighted by geographic proximity to each test point, so intervals in Taunton reflect error behaviour in the South West, not the national average.

## What's in this package

Two sub-systems, one install:

**BYM2 territory ratemaking** (`insurance_spatial` top-level):
- Build adjacency matrices from GeoJSON polygon files or grids
- Fit BYM2 Poisson models via PyMC v5
- Test spatial autocorrelation with Moran's I
- Extract territory relativities with credibility intervals, ready as GLM offsets
- MCMC convergence diagnostics

**Spatially weighted conformal prediction** (`insurance_spatial.conformal`):
- Geographically calibrated prediction intervals for any sklearn-compatible model
- Gaussian, Epanechnikov, and uniform spatial kernels
- Cross-validated bandwidth selection using spatial blocking
- Tweedie Pearson non-conformity scores (recommended for GLM/GBM pricing models)
- MACG (Mean Absolute Coverage Gap) spatial diagnostic
- FCA Consumer Duty table: coverage by geographic region

## Installation

```bash
uv add insurance-spatial
```

With optional geo dependencies (shapefiles and spatial weights):
```bash
uv add "insurance-spatial[geo]"
```

With faster MCMC sampler:
```bash
uv add "insurance-spatial[nutpie]"
```

## BYM2 Quick Start

```python
import numpy as np
from insurance_spatial import build_grid_adjacency, BYM2Model
from insurance_spatial.diagnostics import moran_i

# 1. Build adjacency for a synthetic 10x10 grid of territories.
#    In production, use build_grid_adjacency() from a real postcode-sector grid
#    or AdjacencyMatrix.from_geojson() with ONSPD boundary files.
adj = build_grid_adjacency(10, 10, connectivity="queen")
N = len(adj.areas)  # 100 territories
print(f"Territories: {N}, scaling factor: {adj.scaling_factor:.3f}")

# 2. Synthetic territory data — 100 sectors, true claim rate ~7%.
#    Replace with your actual observed claims and earned exposure per sector.
rng = np.random.default_rng(42)
exposure = rng.uniform(200, 2_000, size=N)          # policy-years per sector
true_log_rate = rng.normal(-2.66, 0.35, size=N)     # spatial variation around 7%
claims = rng.poisson(np.exp(true_log_rate) * exposure)

# 3. Test for spatial autocorrelation before fitting.
#    Run BYM2 only when Moran's I is significant — otherwise simpler
#    credibility weighting suffices and MCMC runtime is wasted.
log_oe = np.log((claims / exposure) / np.mean(claims / exposure) + 1e-8)
test = moran_i(log_oe, adj, n_permutations=999)
print(test.interpretation)

# 4. Fit BYM2 model (requires PyMC — uv add pymc)
model = BYM2Model(adjacency=adj, draws=1000, chains=4)
result = model.fit(
    claims=claims,      # np.ndarray, shape (N,)
    exposure=exposure,  # np.ndarray, shape (N,) — earned policy-years
)

# 5. Check convergence
diag = result.diagnostics()
print(f"Max R-hat: {diag.convergence.max_rhat:.3f}")    # want < 1.01
print(f"Min ESS:   {diag.convergence.min_ess_bulk:.0f}") # want > 400

# 6. Extract territory relativities ready for use as GLM offsets
rels = result.territory_relativities(credibility_interval=0.95)
# area | relativity | lower | upper | ln_offset
# Pass ln_offset as a fixed offset in your downstream Poisson GLM
```

## Spatial Conformal Prediction

Standard conformal prediction guarantees 90% coverage nationally. It does not guarantee 90% coverage in every postcode district. This matters because the FCA expects you to demonstrate fair outcomes across geography, and systematic under-coverage in deprived areas or rural postcodes creates conduct risk.

The spatially weighted conformal predictor wraps your existing fitted model and produces geographically calibrated intervals:

```python
from insurance_spatial.conformal import SpatialConformalPredictor, SpatialCoverageReport


import numpy as np
import polars as pl
from sklearn.dummy import DummyRegressor

# Synthetic UK motor data: 2,000 calibration + 500 test policies with lat/lon
# (UK mainland bounding box: lat 50–58, lon -5–2)
rng = np.random.default_rng(0)
n_cal, n_test = 2_000, 500
lat_cal = rng.uniform(50.5, 57.5, size=n_cal)
lon_cal = rng.uniform(-4.5, 1.5, size=n_cal)
lat_test = rng.uniform(50.5, 57.5, size=n_test)
lon_test = rng.uniform(-4.5, 1.5, size=n_test)
X_cal = rng.standard_normal((n_cal, 5))   # five rating factors
X_test = rng.standard_normal((n_test, 5))
y_cal = rng.gamma(1.5, scale=800, size=n_cal)   # claim severity
# Fit a simple placeholder model (replace with your real CatBoost/LightGBM)
fitted_lgbm = DummyRegressor(strategy='mean').fit(X_cal, y_cal)

# Wrap your fitted GBM or GLM
scp = SpatialConformalPredictor(
    model=fitted_lgbm,
    nonconformity='pearson_tweedie',  # recommended for burning cost models
    tweedie_power=1.5,
    bandwidth_km=20.0,  # or None to select automatically via CV
)

# Calibrate on a held-out set
cal_result = scp.calibrate(
    X_cal, y_cal,
    lat=lat_cal, lon=lon_cal,  # or postcodes=['SW1A 2AA', ...]
)
print(f"Bandwidth: {cal_result.bandwidth_km} km")

# Generate prediction intervals
intervals = scp.predict_interval(
    X_test, lat=lat_test, lon=lon_test, alpha=0.10  # 90% intervals
)
print(intervals.lower[:5], intervals.upper[:5])

# Diagnose spatial coverage quality
report = SpatialCoverageReport(scp)
# Reuse test set as validation; replace with a separate validation split in production
y_test = rng.gamma(1.5, scale=800, size=n_test)   # actual observed values for coverage check
X_val, y_val = X_test, y_test
lat_val, lon_val = lat_test, lon_test
result = report.evaluate(X_val, y_val, lat=lat_val, lon=lon_val)
print(f"MACG: {result.macg:.4f}")  # lower = more spatially uniform coverage

# FCA Consumer Duty table
table = report.fca_consumer_duty_table(region_labels=None)  # pass a list of region names if available
print(table.filter(pl.col('flag') == 'REVIEW'))
```

Using UK postcodes instead of coordinates:

```python
from insurance_spatial.conformal import PostcodeGeocoder

gc = PostcodeGeocoder()
lat_cal, lon_cal = gc.geocode(postcodes_cal)
scp.calibrate(X_cal, y_cal, lat=lat_cal, lon=lon_cal)

# Or pass postcodes directly
scp.calibrate(X_cal, y_cal, postcodes=postcodes_cal)
```

### Non-conformity score choice

The score is the key design decision. For pricing models:

| Score | When to use |
|-------|-------------|
| `pearson_tweedie` (default) | GLM/GBM with Tweedie objective (burning cost, severity) |
| `pearson` | Poisson frequency models |
| `absolute` | Baseline only — ignores heteroscedasticity |
| `scaled_absolute` | Two-model approach with a separate spread model |

The Tweedie Pearson score `|y - yhat| / yhat^(p/2)` variance-stabilises the residuals before weighting, so the spatial kernel is not confounded by the model's own heteroscedasticity.

### Bandwidth selection

If you do not supply `bandwidth_km`, it is selected via spatial blocking cross-validation. The CV minimises MACG (Mean Absolute Coverage Gap) across a spatial grid, which directly measures what matters — geographic coverage consistency.

```python
# Auto bandwidth selection
scp = SpatialConformalPredictor(model=fitted_model, bandwidth_km=None)
result = scp.calibrate(
    X_cal, y_cal, lat=lat_cal, lon=lon_cal,
    cv_candidates_km=[5.0, 10.0, 20.0, 30.0, 50.0],
    cv_folds=5,
)
print(f"CV-selected bandwidth: {result.bandwidth_km} km")
```

## The BYM2 model

The model for area i:

```
y_i ~ Poisson(mu_i)
log(mu_i) = log(E_i) + alpha + X_i @ beta + b_i
b_i = sigma * (sqrt(rho / s) * phi_i + sqrt(1-rho) * theta_i)

phi ~ ICAR(W)           # structured spatial component
theta ~ Normal(0, 1)    # unstructured IID component
sigma ~ HalfNormal(1)   # total territory SD
rho ~ Beta(0.5, 0.5)    # proportion attributable to spatial structure
```

`s` is the BYM2 scaling factor — the geometric mean of the marginal variances of the ICAR precision matrix. It ensures `phi` has unit variance, so `rho` and `sigma` are interpretable regardless of the graph topology.

**Why the rho parameter matters.** After fitting, the posterior of `rho` tells you directly how much of the residual geographic variation is spatially smooth. If `rho → 1`, nearby sectors genuinely tend to have similar risk; BYM2 smoothing is adding real information. If `rho → 0`, territory variation is area-specific noise; the data do not support spatial smoothing and you are better off with simpler credibility weighting.

## Recommended pipeline

The library supports two use patterns:

**Integrated:** pass raw claims and exposure per sector. The model captures all geographic variation.

**Two-stage (recommended for production):** fit your main GLM or GBM without territory, compute sector-level O/E ratios, then pass those to BYM2. This keeps the spatial model decoupled and easier to explain:

```python
# Stage 1: base GLM without territory
# ...compute expected claims per sector from base model...

# Stage 2: spatial model on residual O/E
result = model.fit(
    claims=sector_observed_claims,
    exposure=sector_expected_claims,  # <-- E_i is the base model's fitted value
)
```

The two-stage approach also means the territory factor is auditable independently of the main rating model — useful for regulatory filings.

## UK data sources

To get started with real UK territory data:

| Data | Source | Use |
|---|---|---|
| Postcode sector boundaries | Derived from OS CodePoint Open (free) via Voronoi | Adjacency construction |
| ONSPD | ONS Open Geography Portal | Postcode → sector/LSOA lookup |
| Index of Multiple Deprivation | MHCLG (gov.uk) | Covariates |
| Vehicle crime by LSOA | data.police.uk | Covariates |
| Flood risk by postcode | Environment Agency (data.gov.uk) | Home insurance covariates |

See the demo notebook for a full synthetic example and comments on each data source.

## Computational notes

For N=11,200 UK postcode sectors, the ICAR model is feasible — the pairwise difference formulation is O(N·K) where K≈6 mean neighbours. Published benchmarks suggest ~20–30 minutes for 4 chains × 1,000 draws on modern hardware.

**BYM2 scaling factor: O(N^3) warning.** `compute_bym2_scaling_factor()` uses dense eigendecomposition and is O(N^3). This is fine for the toy grids in the quickstart and for regional models up to N≈3,000 areas. For full UK postcode sector models (N≈9,500), precompute the scaling factor once and pass it as a cached value: `AdjacencyMatrix(_scaling_factor=cached_value)`. The scaling factor depends only on the adjacency graph, not the claims data — recompute it only when the geography is updated, not on every model run.

The scaling factor computation (`adj.scaling_factor`) is a one-off per graph topology; cache it between runs.

nutpie is recommended for production: `uv add nutpie`. It uses a Rust NUTS implementation and is typically 2–5x faster than PyMC's default sampler for models of this type.

---


## Performance

Benchmarked on a synthetic 12×12 grid of postcode sectors (144 areas) with known spatially autocorrelated true rates and heterogeneous exposure. Full script: `benchmarks/run_benchmark.py`.

Three approaches compared: raw observed frequency, quintile banding (5 bands by raw O/E), and BYM2 spatial smoothing (from the larger `benchmarks/benchmark.py` with PyMC).

| Metric | Raw rates | Quintile banding | BYM2 (full benchmark) |
|--------|-----------|-----------------|----------------------|
| MSE vs true rates (overall) | 0.001724 | 0.001055 | lowest |
| MSE vs true rates (thin areas, <30 py) | 0.004048 | 0.001555 | lowest |
| MSE vs true rates (thick areas, ≥100 py) | 0.000480 | 0.000504 | matches thick raw |
| Moran's I on residuals | high | moderate | near zero |
| Fit time | instant | instant | 3–8 min (MCMC) |

On this DGP the Moran's I test returned p=0.21 — not significant at p<0.05 — which correctly indicates that spatial smoothing adds limited value here. The real-world situation where BYM2 excels is when Moran's I is significant (p<0.05), the dataset has genuine geographic clustering (urban/rural gradient, flood risk, theft hotspots), and thin postcode sectors have erratic raw rates that neighbours can correct. Run `benchmarks/benchmark.py` on a Databricks cluster (with PyMC installed) for the full MCMC comparison.

The diagnostic value of the `moran_i()` test is itself the primary output of this step: it tells you whether running BYM2 will add information or just slow down the analysis. If Moran's I p>0.10, use simpler credibility weighting.

**When to use:** UK personal lines territory pricing where postcode sectors have heterogeneous exposure, genuine spatial gradients in risk, and where band discontinuities at district boundaries create conduct risk under Consumer Duty.

**When NOT to use:** When Moran's I is not significant. Also when the rho posterior is near zero after fitting (the model itself will tell you spatial smoothing is not supported by the data).


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/spatial_territory_ratemaking.py).

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [insurance-multilevel](https://github.com/burning-cost/insurance-multilevel) | Broker and scheme random effects — the same credibility-weighting logic applied to group factors instead of territory |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub closed-form credibility — simpler alternative when spatial correlation is not the primary concern |
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract the base model's implicit territory effect before passing O/E ratios to BYM2 |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Test whether a postcode factor is a genuine risk driver or a proxy for a protected characteristic |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion and retention modelling — territory is a key feature in demand models |

[All Burning Cost libraries →](https://burning-cost.github.io)

## References

- Riebler, A., Sørbye, S.H., Simpson, D., & Rue, H. (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
- Gschlössl, S., Schelldorfer, J., & Schnaus, M. (2019). Spatial statistical modelling of insurance risk. *Scandinavian Actuarial Journal*.
- Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with two applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1–40.
- Hjort, N. L., Jullum, M., & Loland, A. (2025). Uncertainty quantification in automated valuation models with spatially weighted conformal prediction. *IJDSA (Springer)*. doi:10.1007/s41060-025-00862-4
- Tibshirani, R. J., Barber, R. F., Candes, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS 2019*.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin rating cells — the same partial-pooling logic applied to non-geographic grouping factors |
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | GLM tooling including R2VF factor merging — use to band territory factors after BYM2 produces the relativities |

## Licence

MIT
