"""
Cross-validated bandwidth selection for spatially weighted conformal prediction.

Choosing the kernel bandwidth is the key tuning decision. Too narrow: insufficient
calibration data near each test point, high variance intervals. Too wide: spatial
weighting becomes uniform, and you've lost the coverage localisation benefit.

The selection procedure uses spatial blocking cross-validation rather than random
folds. Random CV breaks spatial autocorrelation structure: train and validation
points end up centimetres apart, so the CV loss is overly optimistic about
local calibration quality. Spatial blocking groups geographically similar
calibration points into folds.

The objective is MACG (Mean Absolute Coverage Gap) across a spatial grid. This
directly measures what we care about — whether coverage is consistent across
geography — rather than a proxy loss.

A secondary constraint: effective sample size (Kish formula) must be >= n_eff_min
at each test point. In rural Scotland, many bandwidths will fail this constraint;
the selector automatically widens until the floor is met.

References:
    Roberts et al. (2017) 'Cross-validation strategies for data with temporal,
    spatial, hierarchical, or phylogenetic structure.'
    Tibshirani et al. (2019) 'Conformal prediction under covariate shift.'
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from insurance_spatial.conformal._types import BandwidthCVResult
from insurance_spatial.conformal._kernel import haversine_distances, gaussian_weights


class BandwidthSelector:
    """
    Spatially blocked cross-validation for Gaussian kernel bandwidth selection.

    Evaluates candidate bandwidths by (a) creating spatial folds via K-means on
    coordinates, (b) for each fold-as-validation, computing weighted conformal
    quantiles on the training folds, (c) evaluating coverage on the validation fold
    using a spatial grid MACG objective.

    Parameters
    ----------
    candidates_km : list of float, default [2, 5, 10, 15, 20, 30, 50]
        Candidate bandwidths in kilometres. The full list is evaluated; pick based
        on expected UK postcode density. Start with the defaults.
    cv : int, default 5
        Number of spatial folds.
    n_eff_min : int, default 30
        Minimum effective sample size (Kish formula) at each test point.
        Bandwidths producing lower effective N than this are penalised.
    metric : str, default 'macg'
        CV objective. Only 'macg' (mean absolute coverage gap) is currently
        supported.
    grid_resolution : int, default 10
        Grid resolution for MACG calculation during CV (per side). Lower is faster.
    random_state : int, optional
        Random seed for K-means clustering.

    Examples
    --------
    >>> selector = BandwidthSelector(candidates_km=[5, 10, 20, 30], cv=5)
    >>> result = selector.select(
    ...     scores=cal_scores,
    ...     lat=cal_lat,
    ...     lon=cal_lon,
    ...     alpha=0.10,
    ... )
    >>> print(f"Optimal bandwidth: {result.optimal_km} km")
    """

    def __init__(
        self,
        candidates_km: list[float] = None,
        cv: int = 5,
        n_eff_min: int = 30,
        metric: str = "macg",
        grid_resolution: int = 10,
        random_state: Optional[int] = 42,
    ) -> None:
        if candidates_km is None:
            candidates_km = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
        self.candidates_km = sorted(float(c) for c in candidates_km)
        self.cv = int(cv)
        self.n_eff_min = int(n_eff_min)
        self.metric = metric
        self.grid_resolution = int(grid_resolution)
        self.random_state = random_state

        if metric != "macg":
            raise ValueError(f"Only 'macg' metric is currently supported, got '{metric}'")

    def _spatial_folds(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> np.ndarray:
        """
        Assign fold labels via K-means on isotropic projected coordinates.

        Using K-means on coordinates gives spatially contiguous folds,
        preventing data leakage across geographically proximate observations.

        At UK latitudes (~50-60N), raw lon coordinates are compressed relative
        to lat: one degree of longitude is 60-70 km while one degree of latitude
        is ~111 km.  Clustering on (lat, lon) directly creates taller-than-wide
        elliptical folds, which undermines the spatial blocking rationale.

        P1 fix: scale longitude by cos(mean_lat) before clustering.  This makes
        the Euclidean distance in (lat, lon_scaled) space isotropic — one unit
        represents approximately equal real-world distance in both directions.

        Returns
        -------
        np.ndarray of int, shape (n,), values in [0, cv).
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        # Isotropic projection: scale lon so that Euclidean distances approximate
        # real-world distances equally in both dimensions.
        mean_lat_rad = np.deg2rad(lat.mean())
        lon_scaled = lon * np.cos(mean_lat_rad)
        coords = np.column_stack([lat, lon_scaled])
        km = KMeans(
            n_clusters=self.cv,
            n_init=10,
            random_state=self.random_state,
        )
        return km.fit_predict(coords)

    def _kish_n_eff(self, weights: np.ndarray) -> float:
        """
        Kish effective sample size: (sum w)^2 / sum(w^2).

        A measure of how many equally-weighted observations the weighted set
        is equivalent to. Used to flag bandwidths too narrow for sparse areas.
        """
        w = np.asarray(weights)
        sum_w = np.sum(w)
        sum_w2 = np.sum(w ** 2)
        if sum_w2 < 1e-16:
            return 0.0
        return float(sum_w ** 2 / sum_w2)

    def _weighted_quantile(
        self, scores: np.ndarray, weights: np.ndarray, alpha: float
    ) -> float:
        """
        Weighted quantile using the Tibshirani (2019) framework.

        Adds an extra 'infinity' point with weight 1.0 (matching kernel(0) = 1
        for Gaussian kernel at zero distance) for finite-sample validity.
        The (1-alpha) quantile of the augmented weighted distribution.

        Parameters
        ----------
        scores : np.ndarray, shape (n,)
        weights : np.ndarray, shape (n,)
        alpha : float

        Returns
        -------
        float
        """
        n = len(scores)
        # Augment: add infinity with weight 1.0 (Gaussian kernel at zero distance)
        w_aug = np.append(weights, 1.0)
        s_aug = np.append(scores, np.inf)

        # Normalise
        w_total = np.sum(w_aug)
        if w_total < 1e-16:
            return float("inf")
        w_norm = w_aug / w_total

        # Compute weighted CDF and find (1-alpha) quantile
        sort_idx = np.argsort(s_aug)
        s_sorted = s_aug[sort_idx]
        w_sorted = w_norm[sort_idx]
        cdf = np.cumsum(w_sorted)

        idx = np.searchsorted(cdf, 1.0 - alpha)
        if idx >= len(s_sorted):
            return float("inf")
        return float(s_sorted[idx])

    def _macg_on_grid(
        self,
        scores_cal: np.ndarray,
        lat_cal: np.ndarray,
        lon_cal: np.ndarray,
        scores_val: np.ndarray,
        lat_val: np.ndarray,
        lon_val: np.ndarray,
        bandwidth_km: float,
        alpha: float,
    ) -> float:
        """
        Compute MACG for a single bandwidth on hold-out data.

        Creates a spatial grid, computes coverage within each cell,
        and returns the mean absolute deviation from (1-alpha).
        """
        if len(lat_val) == 0:
            return float("nan")

        # Build grid over validation extent
        lat_min, lat_max = lat_val.min(), lat_val.max()
        lon_min, lon_max = lon_val.min(), lon_val.max()

        # Add small buffer
        dlat = max((lat_max - lat_min) * 0.05, 0.01)
        dlon = max((lon_max - lon_min) * 0.05, 0.01)
        lat_edges = np.linspace(lat_min - dlat, lat_max + dlat, self.grid_resolution + 1)
        lon_edges = np.linspace(lon_min - dlon, lon_max + dlon, self.grid_resolution + 1)

        target_coverage = 1.0 - alpha
        gaps = []

        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                # Validation points in this cell
                in_cell = (
                    (lat_val >= lat_edges[i]) & (lat_val < lat_edges[i + 1]) &
                    (lon_val >= lon_edges[j]) & (lon_val < lon_edges[j + 1])
                )
                n_cell = int(np.sum(in_cell))
                if n_cell < 3:
                    continue

                # Cell centre for weighting calibration scores
                cell_lat = 0.5 * (lat_edges[i] + lat_edges[i + 1])
                cell_lon = 0.5 * (lon_edges[j] + lon_edges[j + 1])

                # Weights from calibration to cell centre
                w = gaussian_weights(
                    lat_cal, lon_cal,
                    np.array([cell_lat]), np.array([cell_lon]),
                    bandwidth_km,
                )[:, 0]  # shape (n_cal,)

                q = self._weighted_quantile(scores_cal, w, alpha)

                # Coverage in cell
                cell_scores = scores_val[in_cell]
                cell_covered = float(np.mean(cell_scores <= q))
                gaps.append(abs(target_coverage - cell_covered))

        if len(gaps) == 0:
            return float("nan")
        return float(np.mean(gaps))

    def select(
        self,
        scores: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        alpha: float = 0.10,
    ) -> BandwidthCVResult:
        """
        Select optimal bandwidth via spatially blocked cross-validation.

        Parameters
        ----------
        scores : np.ndarray, shape (n_cal,)
            Pre-computed non-conformity scores on the calibration set.
        lat : np.ndarray, shape (n_cal,)
            Calibration point latitudes.
        lon : np.ndarray, shape (n_cal,)
            Calibration point longitudes.
        alpha : float, default 0.10
            Miscoverage level for coverage evaluation.

        Returns
        -------
        BandwidthCVResult
            Contains optimal_km, cv_scores per candidate, and metadata.

        Examples
        --------
        >>> result = selector.select(cal_scores, cal_lat, cal_lon, alpha=0.10)
        >>> best_bw = result.optimal_km
        """
        scores = np.asarray(scores, dtype=float)
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)

        n = len(scores)
        if n < 2 * self.cv:
            raise ValueError(
                f"Need at least {2 * self.cv} calibration points for {self.cv}-fold CV, "
                f"got {n}."
            )

        fold_labels = self._spatial_folds(lat, lon)
        cv_scores_by_bw: dict[float, list[float]] = {bw: [] for bw in self.candidates_km}

        for fold in range(self.cv):
            is_val = fold_labels == fold
            is_cal = ~is_val

            s_cal = scores[is_cal]
            lat_cal_f = lat[is_cal]
            lon_cal_f = lon[is_cal]
            s_val = scores[is_val]
            lat_val_f = lat[is_val]
            lon_val_f = lon[is_val]

            for bw in self.candidates_km:
                macg = self._macg_on_grid(
                    s_cal, lat_cal_f, lon_cal_f,
                    s_val, lat_val_f, lon_val_f,
                    bw, alpha,
                )
                cv_scores_by_bw[bw].append(macg)

        # Mean CV MACG per bandwidth (ignoring NaN folds)
        mean_cv = {}
        for bw in self.candidates_km:
            vals = [v for v in cv_scores_by_bw[bw] if not np.isnan(v)]
            mean_cv[bw] = float(np.mean(vals)) if vals else float("inf")

        optimal_km = min(mean_cv, key=lambda b: mean_cv[b])
        cv_score_list = [mean_cv[bw] for bw in self.candidates_km]

        return BandwidthCVResult(
            optimal_km=optimal_km,
            candidates_km=self.candidates_km,
            cv_scores=cv_score_list,
            metric=self.metric,
            n_folds=self.cv,
        )

    def select_with_n_eff_floor(
        self,
        scores: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        test_lat: np.ndarray,
        test_lon: np.ndarray,
        alpha: float = 0.10,
    ) -> BandwidthCVResult:
        """
        Select bandwidth, but auto-widen if effective N falls below n_eff_min.

        Useful for rural portfolios where the optimal CV bandwidth might be too
        narrow to achieve reliable coverage estimates.

        Parameters
        ----------
        scores : np.ndarray
        lat : np.ndarray
        lon : np.ndarray
        test_lat : np.ndarray
            Test point latitudes (used to check effective N).
        test_lon : np.ndarray
            Test point longitudes.
        alpha : float, default 0.10

        Returns
        -------
        BandwidthCVResult
            The result has optimal_km set to the widened bandwidth if needed.
            When widened, optimal_km is updated in-place on the returned object.
            Note: cv_scores in the result correspond to the original CV run and
            do not reflect the widened bandwidth's CV score (which was not computed).
        """
        result = self.select(scores, lat, lon, alpha)

        # Check n_eff at test points
        bw = result.optimal_km
        w_matrix = gaussian_weights(lat, lon, test_lat, test_lon, bw)
        n_eff_per_test = np.array([
            self._kish_n_eff(w_matrix[:, j]) for j in range(len(test_lat))
        ])
        min_n_eff = float(np.min(n_eff_per_test))

        if min_n_eff < self.n_eff_min:
            # Find the widest candidate with sufficient n_eff at all test points.
            # Sorts in descending order (widest first) so we return the first
            # that satisfies the floor — i.e. the widest adequate bandwidth,
            # which is the most conservative choice consistent with the constraint.
            for candidate_bw in sorted(self.candidates_km, reverse=True):
                w_mat = gaussian_weights(lat, lon, test_lat, test_lon, candidate_bw)
                n_eff_vals = np.array([
                    self._kish_n_eff(w_mat[:, j]) for j in range(len(test_lat))
                ])
                if float(np.min(n_eff_vals)) >= self.n_eff_min:
                    warnings.warn(
                        f"CV-optimal bandwidth ({bw} km) gives min effective N = "
                        f"{min_n_eff:.1f} < {self.n_eff_min}. "
                        f"Widening to {candidate_bw} km.",
                        UserWarning,
                        stacklevel=2,
                    )
                    result.optimal_km = candidate_bw
                    return result

            warnings.warn(
                f"No candidate bandwidth achieves n_eff >= {self.n_eff_min} at all "
                "test points. Returning CV-optimal bandwidth regardless.",
                UserWarning,
                stacklevel=2,
            )

        return result
