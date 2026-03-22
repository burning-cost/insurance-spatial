# Changelog

## v0.2.3 (2026-03-22) [unreleased]
- Add Databricks benchmark: BYM2 vs GLM dummies vs naive geographic mean
- fix: use plain string license field for universal setuptools compatibility
- fix: use license text instead of file reference in pyproject.toml
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.2.3 (2026-03-21)
- Add blog post link and community CTA to README
- Add benchmarks/run_benchmark.py and update Performance section
- QA batch 9 fixes: Moran's I p-value, scaling factor warning, version bump
- Fix P0/P1 bugs from v0.2.1 code review; bump to v0.2.2
- Add standalone benchmark script
- fix: lazy-import conformal sub-package to avoid eager sklearn import
- docs: add Databricks notebook link
- fix: pin arviz<1.0 to prevent incompatibility with pymc 5.28+
- Add Related Libraries section to README
- fix: README conformal example used model predictions as observed values
- fix: update cross-references to consolidated repos
- fix: make BYM2 and conformal quick-start blocks self-contained
- fix: update polars floor to >=1.0 and fix project URLs
- Add Performance section to README
- Add benchmark notebook: BYM2 spatial smoothing vs territory banding
- Absorb insurance-spatial-conformal as insurance_spatial.conformal sub-package
- Move PyMC/pytensor to optional [mcmc] extra, add __version__ to __all__

## v0.1.0 (2026-03-09)
- fix: SyntaxError unescaped quotes in adjacency.py; add pyarrow+pymc to dev
- docs: add blog link, Related libraries, fix broken org URLs, standardise headings
- Add GitHub Actions CI workflow and test badge
- fix: update URLs to burning-cost org
- Add badges and cross-links to README
- Replace pip install / uv pip install with uv add throughout
- docs: switch examples to CatBoost/polars/uv, fix tone
- fix: standardise on CatBoost, uv, clean up style
- fix: uv references
- Fix BYM2 scaling factor: use proper eigendecomposition of ICAR Laplacian
- Fix test_reasonable_magnitude: ICAR scaling factor is O(N) for grid graphs
- Use dbutils.notebook.exit to capture test output in run result
- Fix test notebook: add PYTHONPATH and print full output before raising
- Add Databricks test runner notebooks
- Initial release: BYM2 spatial territory ratemaking library

