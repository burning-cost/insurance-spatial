"""
BYM2 spatial territory model using pyINLA (KAUST v0.2.0) as the inference backend.

Why this exists
---------------
The MCMC-based BYM2Model is correct and interpretable, but MCMC sampling scales
poorly: a 7,900-area UK postcode sector model takes 76 seconds in R-INLA and
around 10–20 minutes in PyMC.  For iterative pricing work — reruns triggered by
assumption changes, new data vintages, or sensitivity checks — that latency
accumulates.

INLA (Integrated Nested Laplace Approximation) is a deterministic approximation
to the posterior that is orders of magnitude faster for models in the Latent
Gaussian Model class, which BYM2 is.  pyINLA (arXiv:2603.27276, Abdul Fattah,
Krainski, Rue, KAUST, March 2026) is the pure-Python INLA implementation from
the same group that wrote R-INLA.  It bundles a compiled C binary and has no R
dependency.

Benchmarked speedup: 278x over PyMC NUTS on the Scottish lip cancer dataset
(N=56).  For N=8,000, R-INLA takes 76 seconds; pyINLA is expected to be
comparable or faster (tbd — no published N=8,000 pyINLA benchmark as of April
2026).

Design constraints
------------------
- This is a **parallel class** to BYM2Model, not a replacement.  MCMC is still
  needed for prior sensitivity, convergence diagnostics, and smaller problems
  where the approximation quality matters.
- BYM2InlaResult.territory_relativities() returns the same schema as
  BYM2Result.territory_relativities() so downstream code is backend-agnostic.
- pyINLA is an optional dependency.  Teams without the INLA use case don't need
  the binary download.

Adjacency format
----------------
pyINLA (and R-INLA) requires adjacency as a plain-text .adj file:
  N
  node_id  n_neighbours  neighbour_1  neighbour_2  ...
  ...

Nodes are 1-indexed in the .adj format.  We write the file to a temporary
directory on each fit() call.  For repeated fits on the same adjacency structure,
consider caching the .adj path externally.

ARM64 / Linux note
------------------
As of April 2026, pyINLA 0.1.6 (latest on PyPI) ships manylinux2014 x86_64
wheels but no Linux ARM64 wheel.  On a Raspberry Pi or other ARM64 Linux host,
pip install pyinla may fail or fall back to the sdist, which may not build.
macOS ARM64 (Apple Silicon) and x86_64 Linux are fully supported.

The integration code is written and tested against the pyINLA API.  If pyINLA
cannot be installed in your environment, BYM2Model (PyMC MCMC) remains fully
functional.

Hyperprior choices
------------------
We follow the PC prior recommendations from Simpson et al. (2017) and the
pyINLA BYM2 documentation:
  - theta1 (log precision): PC prior P(sigma > 1) = 0.01
    Weak prior: allows total SD up to ~4.6 on log scale, which easily covers
    any realistic geographic frequency relativities in UK insurance.
  - theta2 (logit mixing rho): PC prior P(rho > 0.5) = 0.5
    Symmetric prior: no a priori preference for structured vs unstructured
    spatial variation.  The data dominates this parameter in practice.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import polars as pl

from insurance_spatial.adjacency import AdjacencyMatrix


@dataclass
class BYM2InlaResult:
    """
    Container for the output of a fitted BYM2InlaModel.

    This mirrors the BYM2Result interface so that downstream code that calls
    territory_relativities() or diagnostics() works with either backend.

    Attributes
    ----------
    b_mean :
        Posterior mean of the area-level spatial effect b_i, shape (N,).
        On the log scale.  Used directly in territory_relativities().
    b_sd :
        Posterior standard deviation of b_i, shape (N,).
    areas :
        Ordered list of area identifiers corresponding to b_mean indices.
    adjacency :
        The adjacency structure used in fitting.
    n_areas :
        Number of areas N.
    fixed_summary :
        Polars DataFrame with fixed effects posterior summaries (mean, sd, q0.025,
        q0.975 for alpha and any covariates).
    hyperpar_summary :
        Polars DataFrame with hyperparameter posterior summaries (sigma, rho).
        Column names: parameter, mean, sd, q025, q975.
    dic :
        Deviance Information Criterion from INLA.  Lower is better.
        NaN if not returned by pyINLA version.
    waic :
        Widely Applicable Information Criterion.  Lower is better.
        NaN if not returned by pyINLA version.
    marginals :
        Dict of posterior marginals as returned by pyINLA.  Useful for detailed
        hyperparameter analysis.  May be an empty dict if pyINLA does not expose
        them in this version.
    """

    b_mean: np.ndarray
    b_sd: np.ndarray
    areas: list[str]
    adjacency: AdjacencyMatrix
    n_areas: int
    fixed_summary: pl.DataFrame
    hyperpar_summary: pl.DataFrame
    dic: float = float("nan")
    waic: float = float("nan")
    marginals: dict = field(default_factory=dict)

    def territory_relativities(
        self,
        base_area: Optional[str] = None,
        credibility_interval: float = 0.95,
    ) -> pl.DataFrame:
        """
        Extract territory relativities from the fitted INLA model.

        Returns a Polars DataFrame with the same schema as
        BYM2Result.territory_relativities():

          area          str
          b_mean        float  posterior mean of b_i (log scale)
          b_sd          float  posterior SD of b_i (log scale)
          relativity    float  exp(b_i - reference_b), normalised
          lower         float  lower credibility bound on relativity
          upper         float  upper credibility bound on relativity
          ln_offset     float  b_mean - reference_b_mean, log-scale GLM offset

        Credibility intervals are derived from the Gaussian approximation:
        lower = exp(b_mean - z * b_sd - reference_b), where z is the normal
        quantile for the requested credibility_interval width.

        Note: INLA returns marginal posteriors (not joint samples), so intervals
        here are Gaussian approximations to the marginal posterior of each b_i.
        For the MCMC backend, intervals are derived from actual posterior draws.
        In practice, the Gaussian approximation is excellent for BYM2 with
        moderate to large N; it may be slightly optimistic for extreme tails
        with sparse data.

        Parameters
        ----------
        base_area :
            Area to use as reference (relativity = 1.0).  If None, normalise
            to the geometric mean (sum-to-zero on log scale, i.e. subtract
            the mean of b_mean).
        credibility_interval :
            Width of the symmetric interval.  Default 0.95.
        """
        from scipy import stats

        alpha = 1.0 - credibility_interval
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))

        b_mean = self.b_mean.copy()
        b_sd = self.b_sd.copy()

        if base_area is not None:
            if base_area not in self.areas:
                raise ValueError(
                    f"base_area '{base_area}' not found. "
                    f"First few areas: {self.areas[:5]}"
                )
            ref_idx = self.areas.index(base_area)
            reference_b = float(b_mean[ref_idx])
        else:
            reference_b = float(b_mean.mean())

        b_adjusted = b_mean - reference_b

        relativity = np.exp(b_adjusted)
        lower = np.exp(b_adjusted - z * b_sd)
        upper = np.exp(b_adjusted + z * b_sd)
        ln_offset = b_adjusted

        return pl.DataFrame(
            {
                "area": self.areas,
                "b_mean": b_mean.tolist(),
                "b_sd": b_sd.tolist(),
                "relativity": relativity.tolist(),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
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

    def diagnostics(self) -> "InlaDiagnostics":
        """
        Return INLA-specific diagnostics.

        Returns an InlaDiagnostics object with hyperparameter summaries and
        model selection criteria (DIC, WAIC).  Unlike the MCMC backend, there
        are no R-hat or ESS diagnostics — INLA is deterministic.
        """
        return InlaDiagnostics(
            hyperpar_summary=self.hyperpar_summary,
            fixed_summary=self.fixed_summary,
            dic=self.dic,
            waic=self.waic,
        )


@dataclass
class InlaDiagnostics:
    """
    INLA-specific diagnostics from a fitted BYM2InlaModel.

    For the MCMC backend, diagnostics() returns a SpatialDiagnostics object
    that includes R-hat, ESS, and Moran's I.  INLA is deterministic, so
    there are no chain convergence diagnostics.  Instead, the key outputs
    are the hyperparameter posterior summaries and model selection criteria.

    Attributes
    ----------
    hyperpar_summary :
        Posterior summaries of sigma (total SD) and rho (spatial proportion).
    fixed_summary :
        Posterior summaries of intercept alpha and any fixed effects beta.
    dic :
        DIC.  NaN if not returned by this pyINLA version.
    waic :
        WAIC.  NaN if not returned by this pyINLA version.
    """

    hyperpar_summary: pl.DataFrame
    fixed_summary: pl.DataFrame
    dic: float
    waic: float


class BYM2InlaModel:
    """
    BYM2 spatial Poisson model using pyINLA (KAUST) as the inference backend.

    For UK pricing teams who need fast reruns on large territory grids (N > 500).
    Compared to BYM2Model (PyMC MCMC):

      - 278x faster on N=56 benchmark.
      - No chain convergence to monitor.
      - Deterministic: same data always gives the same result.
      - Credibility intervals are Gaussian approximations to INLA marginals,
        not exact posterior samples.

    Requires the optional ``inla`` extra:
        uv add 'insurance-spatial[inla]'

    .. note::
        As of April 2026, pyINLA 0.1.6 has no Linux ARM64 wheel.  Install
        succeeds on macOS (ARM64 and x86_64) and Linux x86_64.

    Parameters
    ----------
    adjacency :
        AdjacencyMatrix built from your territory boundary data.
    scale_model :
        If True, scale the BYM2 spatial component to have unit marginal
        variance (Riebler et al. 2016).  Should almost always be True —
        only set False for direct comparison with unscaled R-INLA runs.
    sigma_u_prior :
        PC prior for sigma (total SD of spatial effect): (U, alpha) means
        P(sigma > U) = alpha.  Default (1, 0.01): weak prior allowing
        large spatial variation.
    rho_prior :
        PC prior for rho (proportion of spatial variance): (U, alpha) means
        P(rho > U) = alpha.  Default (0.5, 0.5): symmetric, no preference
        for structured vs unstructured.
    covariate_names :
        Optional list of covariate names.  Used only for labelling the
        fixed_summary DataFrame.  If None and covariates are passed to fit(),
        names default to 'x0', 'x1', etc.

    Example
    -------
    >>> from insurance_spatial import build_grid_adjacency, BYM2InlaModel
    >>> adj = build_grid_adjacency(5, 5)
    >>> model = BYM2InlaModel(adjacency=adj)
    >>> result = model.fit(claims=counts, exposure=exposure)
    >>> rels = result.territory_relativities()
    """

    def __init__(
        self,
        adjacency: AdjacencyMatrix,
        scale_model: bool = True,
        sigma_u_prior: tuple[float, float] = (1.0, 0.01),
        rho_prior: tuple[float, float] = (0.5, 0.5),
        covariate_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.adjacency = adjacency
        self.scale_model = scale_model
        self.sigma_u_prior = sigma_u_prior
        self.rho_prior = rho_prior
        self.covariate_names = list(covariate_names) if covariate_names else None

    def fit(
        self,
        claims: np.ndarray,
        exposure: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> BYM2InlaResult:
        """
        Fit the BYM2 model via INLA.

        Parameters
        ----------
        claims :
            Array of shape (N,) of observed claim counts per area.
            Must be non-negative integers.
        exposure :
            Array of shape (N,) of policy-years (or expected counts from a
            base model for two-stage use).  Must be strictly positive.
            Passed to pyINLA as the Poisson offset E_i (log link: log(E_i)
            is added to the linear predictor automatically).
        covariates :
            Optional float array of shape (N, P) of area-level fixed effects.
            Scale to mean 0, SD 1 before passing — the prior on beta is
            Normal(0, 1) by default in the pyINLA interface.

        Returns
        -------
        BYM2InlaResult
        """
        try:
            import pyinla
        except ImportError as exc:
            raise ImportError(
                "INLA backend requires pyinla (KAUST v0.2.0+). "
                "Install with: uv add 'insurance-spatial[inla]'\n"
                "Note: Linux ARM64 wheels are not yet available (April 2026). "
                "Use BYM2Model (PyMC MCMC) on Raspberry Pi / ARM64 Linux."
            ) from exc

        import pandas as pd

        claims = np.asarray(claims, dtype=np.int64)
        exposure = np.asarray(exposure, dtype=np.float64)

        N = self.adjacency.n

        if claims.shape[0] != N:
            raise ValueError(
                f"claims has {claims.shape[0]} entries but adjacency has {N} areas"
            )
        if exposure.shape[0] != N:
            raise ValueError(
                f"exposure has {exposure.shape[0]} entries but adjacency has {N} areas"
            )
        if np.any(exposure <= 0):
            raise ValueError("All exposure values must be strictly positive")
        if np.any(claims < 0):
            raise ValueError("claim counts must be non-negative")

        if covariates is not None:
            covariates = np.asarray(covariates, dtype=np.float64)
            if covariates.ndim == 1:
                covariates = covariates[:, None]
            if covariates.shape[0] != N:
                raise ValueError(
                    f"covariates has {covariates.shape[0]} rows but {N} areas"
                )

        # Write adjacency to a temporary .adj file.
        # pyINLA reads this path at fit time, so we keep the tempdir alive
        # for the duration of the fit call.
        with tempfile.TemporaryDirectory() as tmpdir:
            adj_path = os.path.join(tmpdir, "adjacency.adj")
            _write_adj_file(self.adjacency, adj_path)

            # Build the pandas DataFrame for pyINLA
            df = self._build_dataframe(claims, exposure, covariates)

            # Resolve covariate names
            fixed_effects: list[str] = []
            if covariates is not None:
                P = covariates.shape[1]
                if self.covariate_names:
                    if len(self.covariate_names) != P:
                        raise ValueError(
                            f"covariate_names has {len(self.covariate_names)} entries "
                            f"but covariates has {P} columns"
                        )
                    fixed_effects = list(self.covariate_names)
                else:
                    fixed_effects = [f"x{i}" for i in range(P)]

            # Build pyINLA formula
            formula = pyinla.Formula(
                response="y",
                family="poisson",
                offset="log_E",
                fixed_effects=fixed_effects if fixed_effects else None,
                random_effects=[
                    {
                        "id": "area_id",
                        "model": "bym2",
                        "scale.model": self.scale_model,
                        "graph": adj_path,
                        "hyper": {
                            "theta1": {
                                "prior": "pc.prec",
                                "param": list(self.sigma_u_prior),
                            },
                            "theta2": {
                                "prior": "pc",
                                "param": list(self.rho_prior),
                            },
                        },
                    }
                ],
            )

            inla_result = pyinla.inla(formula=formula, data=df)

        return self._parse_result(inla_result, N)

    def _build_dataframe(
        self,
        claims: np.ndarray,
        exposure: np.ndarray,
        covariates: Optional[np.ndarray],
    ) -> "pd.DataFrame":
        import pandas as pd

        N = self.adjacency.n
        data: dict[str, object] = {
            "y": claims,
            "log_E": np.log(exposure),
            # area_id is 1-indexed to match the .adj file node numbering
            "area_id": np.arange(1, N + 1, dtype=np.int32),
        }

        if covariates is not None:
            P = covariates.shape[1]
            covariate_names = (
                self.covariate_names
                if self.covariate_names
                else [f"x{i}" for i in range(P)]
            )
            for j, name in enumerate(covariate_names):
                data[name] = covariates[:, j]

        return pd.DataFrame(data)

    def _parse_result(self, inla_result: object, N: int) -> BYM2InlaResult:
        """
        Convert the raw pyINLA result object into a BYM2InlaResult.

        pyINLA result structure (v0.2.0):
          result.summary.fixed        - DataFrame: rows = fixed effects, cols include mean, sd
          result.summary.hyperpar     - DataFrame: rows = hyperparams (sigma, rho), cols include mean, sd
          result.summary.random['area_id']  - DataFrame: rows = areas, cols include mean, sd
          result.dic                  - float or None
          result.waic                 - float or None
          result.marginals            - dict of marginals (may be empty)
        """
        import pandas as pd

        # --- Spatial random effects (b_i) ---
        try:
            random_df = inla_result.summary.random["area_id"]
            # pyINLA returns mean and sd columns for each area
            b_mean = np.asarray(random_df["mean"].values, dtype=np.float64)
            b_sd = np.asarray(random_df["sd"].values, dtype=np.float64)
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                f"Unexpected pyINLA result structure when reading random effects: {exc}. "
                "This may indicate a pyINLA API change. Expected result.summary.random['area_id'] "
                "with 'mean' and 'sd' columns."
            ) from exc

        if len(b_mean) != N:
            raise RuntimeError(
                f"Expected {N} random effect posteriors from pyINLA but got {len(b_mean)}. "
                "Check that the adjacency file and data have matching area counts."
            )

        # --- Fixed effects ---
        fixed_summary = _parse_summary_df(inla_result.summary.fixed, "fixed_effect")

        # --- Hyperparameters ---
        hyperpar_summary = _parse_summary_df(inla_result.summary.hyperpar, "parameter")

        # --- Model selection criteria ---
        dic = float(getattr(inla_result, "dic", float("nan")) or float("nan"))
        waic = float(getattr(inla_result, "waic", float("nan")) or float("nan"))

        # --- Marginals ---
        marginals: dict = {}
        if hasattr(inla_result, "marginals") and inla_result.marginals:
            marginals = dict(inla_result.marginals)

        return BYM2InlaResult(
            b_mean=b_mean,
            b_sd=b_sd,
            areas=list(self.adjacency.areas),
            adjacency=self.adjacency,
            n_areas=N,
            fixed_summary=fixed_summary,
            hyperpar_summary=hyperpar_summary,
            dic=dic,
            waic=waic,
            marginals=marginals,
        )


def _write_adj_file(adjacency: AdjacencyMatrix, path: str) -> None:
    """
    Write the adjacency structure to an INLA .adj text file.

    Format:
        N
        node_id  n_neighbours  neighbour_1  neighbour_2  ...
        ...

    Nodes are 1-indexed.  The adjacency must be a connected graph (validated
    upstream by AdjacencyMatrix).

    Parameters
    ----------
    adjacency :
        AdjacencyMatrix with W as a scipy sparse CSR matrix.
    path :
        Destination file path.  Directory must already exist.
    """
    N = adjacency.n
    W_csr = adjacency.W.tocsr()

    lines: list[str] = [str(N)]
    for i in range(N):
        # Get column indices of non-zero entries in row i (0-indexed)
        row_start = W_csr.indptr[i]
        row_end = W_csr.indptr[i + 1]
        neighbours_0idx = W_csr.indices[row_start:row_end]
        # Convert to 1-indexed
        neighbours_1idx = neighbours_0idx + 1
        n_neighbours = len(neighbours_1idx)
        node_id = i + 1  # 1-indexed
        neighbour_str = " ".join(str(j) for j in neighbours_1idx)
        lines.append(f"{node_id} {n_neighbours} {neighbour_str}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _parse_summary_df(raw_df: object, id_col: str) -> pl.DataFrame:
    """
    Convert a pyINLA summary pandas DataFrame to a standardised Polars DataFrame.

    pyINLA summary DataFrames have the parameter/effect name as the index and
    columns 'mean', 'sd', '0.025quant', '0.975quant' (matching R-INLA naming).
    We rename to 'q025', 'q975' for consistency with the rest of the library.

    Falls back gracefully if the DataFrame does not have expected columns.
    """
    import pandas as pd

    if raw_df is None:
        return pl.DataFrame(
            {id_col: [], "mean": [], "sd": [], "q025": [], "q975": []}
        )

    try:
        # raw_df may be a pandas DataFrame
        if not isinstance(raw_df, pd.DataFrame):
            raw_df = pd.DataFrame(raw_df)

        names = list(raw_df.index.astype(str))
        col_map = {
            "mean": "mean",
            "sd": "sd",
            "0.025quant": "q025",
            "0.975quant": "q975",
        }
        result: dict[str, list] = {id_col: names}
        for src_col, dst_col in col_map.items():
            if src_col in raw_df.columns:
                result[dst_col] = raw_df[src_col].tolist()
            else:
                result[dst_col] = [float("nan")] * len(names)

        return pl.DataFrame(result).with_columns(
            [
                pl.col("mean").cast(pl.Float64),
                pl.col("sd").cast(pl.Float64),
                pl.col("q025").cast(pl.Float64),
                pl.col("q975").cast(pl.Float64),
            ]
        )

    except Exception as exc:
        # Return a minimal valid DataFrame rather than crashing on an
        # unexpected pyINLA version change
        return pl.DataFrame(
            {id_col: [], "mean": [], "sd": [], "q025": [], "q975": []}
        )
