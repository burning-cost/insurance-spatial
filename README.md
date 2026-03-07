# insurance-spatial

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-spatial)

BYM2 spatial territory ratemaking for UK personal lines insurance.

## The Problem

Territory pricing in UK personal lines is broken in predictable ways.

The standard approach - a GLM with postcode sector as a categorical predictor - creates 11,200 separate territory parameters for motor, most of them estimated from a handful of claims. Adjacent sectors can differ by 30–40% on sparse data not because the underlying risk differs but because the estimates are noisy. Standard practice is to band sectors into 6–20 groups using k-means on historical loss ratios. This is ad hoc, discards information, and creates artificial discontinuities at band boundaries.

GBMs handle territory implicitly but produce an uninterpretable spatial effect diffused across hundreds of splits. You cannot extract a territory factor for regulatory filing or actuarial peer review.

Vendor tools (Emblem, Akur8) have some form of spatial smoothing, but the methodology is proprietary. When the FCA asks how your territory factors are derived, "the platform did it" is not a satisfying answer.

The academically-grounded alternative is the **BYM2 model** (Besag-York-Mollié, Riebler et al. 2016): a Bayesian hierarchical model that borrows strength across neighbouring postcode sectors, quantifies how much geographic variation is genuinely spatial vs. idiosyncratic noise, and produces territory relativities with proper uncertainty estimates.

This library wraps that model for UK insurance use.

## What it does

- **Builds adjacency matrices** from GeoJSON polygon files or simple grids
- **Fits BYM2 Poisson models** via PyMC v5's `pm.ICAR` - the structured spatial component captures smooth geographic variation; the IID component captures area-specific outliers
- **Tests for spatial autocorrelation** using Moran's I before and after fitting
- **Extracts territory relativities** as multiplicative factors with credibility intervals, ready to use as GLM offsets
- **Reports convergence** - R-hat, ESS, divergences - because MCMC without diagnostics is not production-ready

## Install

```bash
uv add insurance-spatial
```

With optional geo dependencies (for loading shapefiles and spatial weights):
```bash
uv add "insurance-spatial[geo]"
```

With faster sampler:
```bash
uv add "insurance-spatial[nutpie]"
```

## Quickstart

```python
from insurance_spatial import build_grid_adjacency, BYM2Model
from insurance_spatial.diagnostics import moran_i

# 1. Build adjacency (synthetic grid - use from_geojson() for real data)
adj = build_grid_adjacency(10, 10, connectivity="queen")
print(f"Scaling factor: {adj.scaling_factor:.3f}")

# 2. Test for spatial autocorrelation before fitting
log_oe = ...  # log(observed / expected) per sector, shape (N,)
test = moran_i(log_oe, adj, n_permutations=999)
print(test.interpretation)

# 3. Fit BYM2 model
model = BYM2Model(adjacency=adj, draws=1000, chains=4)
result = model.fit(
    claims=claims,      # np.ndarray, shape (N,)
    exposure=exposure,  # np.ndarray, shape (N,)
)

# 4. Check convergence
diag = result.diagnostics()
print(f"Max R-hat: {diag.convergence.max_rhat:.3f}")    # want < 1.01
print(f"Min ESS:   {diag.convergence.min_ess_bulk:.0f}") # want > 400
print(diag.rho_summary)    # how much variation is spatially structured?

# 5. Extract relativities
rels = result.territory_relativities(credibility_interval=0.95)
# area | relativity | lower | upper | ln_offset
# Use ln_offset as a fixed offset in your downstream GLM
```

Loading real sector boundaries:

```python
from insurance_spatial.adjacency import from_geojson

adj = from_geojson(
    "postcode_sectors.geojson",
    area_col="PC_SECTOR",
    connectivity="queen",
    fix_islands=True,  # connect Scottish islands to nearest mainland sector
)
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

`s` is the BYM2 scaling factor - the geometric mean of the marginal variances of the ICAR precision matrix. It ensures `phi` has unit variance, so `rho` and `sigma` are interpretable regardless of the graph topology.

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

The two-stage approach also means the territory factor is auditable independently of the main rating model - useful for regulatory filings.

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

For N=11,200 UK postcode sectors, the ICAR model is feasible - the pairwise difference formulation is O(N·K) where K≈6 mean neighbours. Published benchmarks suggest ~20–30 minutes for 4 chains × 1,000 draws on modern hardware. The scaling factor computation (`adj.scaling_factor`) is a one-off sparse linear solve per graph topology; cache it between runs.

For exploratory work on district-level data (N≈3,000), a full run takes under 10 minutes.

nutpie is recommended for production: `uv add nutpie`. It uses a Rust NUTS implementation and is typically 2–5x faster than PyMC's default sampler for models of this type.

## References

- Riebler, A., Sørbye, S.H., Simpson, D., & Rue, H. (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
- Gschlössl, S., Schelldorfer, J., & Schnaus, M. (2019). Spatial statistical modelling of insurance risk. *Scandinavian Actuarial Journal*.
- Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with two applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1–20.
- Brockman, M.J., & Wright, T.S. (1992). Statistical motor rating: making effective use of your data. *Journal of the Institute of Actuaries*, 119, 457–543.

---

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burningcost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burningcost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burningcost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burningcost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burningcost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [credibility](https://github.com/burningcost/credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [rate-optimiser](https://github.com/burningcost/rate-optimiser) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burningcost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burningcost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-causal](https://github.com/burningcost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-monitoring](https://github.com/burningcost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries →](https://burningcost.github.io)

---

## Licence

MIT
