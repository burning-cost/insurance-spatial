"""
SpatialConformalPredictor: spatially weighted conformal prediction intervals for insurance.

Standard split conformal prediction gives a guarantee of the form:
    P(y_test in [lo, hi]) >= 1 - alpha

Nationally. But this marginal guarantee can break badly at a local level —
postcodes in the south-west might be systematically over-covered while inner
London is under-covered. That's the exchangeability assumption failing: homes in
Taunton and homes in Hackney have different loss distributions, and treating all
calibration scores as interchangeable doesn't respect this.

The fix is to weight calibration scores by proximity. A test point in Taunton
should lean on calibration data from Taunton and Somerset rather than from the
whole of England. The Gaussian kernel provides a principled way to do this, with
bandwidth controlling how far the influence extends.

The implementation follows Hjort, Jullum, Loland (2025) arXiv:2312.06531 for the
spatial weighting framework, and uses the Tibshirani (2019) weighted conformal
framework for finite-sample validity.

Finite-sample coverage guarantee:
    P(y_test in C(x_test)) >= 1 - alpha

holds marginally, and spatial weighting substantially improves conditional coverage
at geographic granularity.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np

from insurance_spatial.conformal._kernel import compute_weights, kish_n_eff
from insurance_spatial.conformal._types import (
    CalibrationResult,
    IntervalResult,
    KernelType,
    NonconformityType,
)
from insurance_spatial.conformal.scores import make_score


class SpatialConformalPredictor:
    """
    Spatially weighted conformal prediction intervals for insurance pricing models.

    Wraps any fitted sklearn-compatible model and produces geographically calibrated
    prediction intervals. The key innovation over standard conformal prediction is
    the Gaussian spatial kernel: calibration non-conformity scores are weighted by
    their proximity to each test point, so the quantile used for the interval
    reflects local error behaviour rather than national average behaviour.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Must implement predict(X). For exposure-based models (Poisson frequency),
        ensure predict() returns expected counts with exposure already applied.
    nonconformity : str, default 'pearson_tweedie'
        Non-conformity score type. Options:
        - 'pearson_tweedie': |y - yhat| / yhat^(p/2)  [recommended for GLM/GBM]
        - 'pearson': |y - yhat| / sqrt(yhat)  [Poisson frequency]
        - 'absolute': |y - yhat|  [baseline, not recommended for insurance]
        - 'scaled_absolute': |y - yhat| / sigma(X)  [requires spread_model]
    tweedie_power : float, default 1.5
        Tweedie variance power for 'pearson_tweedie' score. Common values:
        1.0 (Poisson), 1.5 (compound Poisson-Gamma, typical burning cost), 2.0 (Gamma).
    spatial_kernel : str, default 'gaussian'
        Kernel for geographic weighting. One of:
        'gaussian' — recommended, smooth decay
        'epanechnikov' — compact support, slightly sharper cutoff
        'uniform' — binary within/outside radius (equivalent to NNCP)
    bandwidth_km : float or None, default None
        Gaussian kernel bandwidth in kilometres. If None, bandwidth is selected
        automatically via cross-validation (BandwidthSelector) during calibrate().
        If float, this fixed bandwidth is used for all predictions.
    spread_model : fitted model, optional
        Required when nonconformity='scaled_absolute'. A model predicting |y - yhat|.
    n_eff_min : int, default 30
        Minimum effective sample size (Kish formula) at each test point.
        A warning is issued when a test point falls below this threshold.
        The predictor still produces an interval; it's just flagged as less reliable.

    Attributes
    ----------
    cal_scores_ : np.ndarray
        Non-conformity scores on the calibration set. Set after calibrate().
    cal_lat_ : np.ndarray
        Calibration point latitudes.
    cal_lon_ : np.ndarray
        Calibration point longitudes.
    bandwidth_km_ : float
        Bandwidth used (either supplied or CV-selected). Set after calibrate().
    is_calibrated_ : bool
        True after calibrate() has been called.

    Examples
    --------
    >>> from insurance_spatial.conformal import SpatialConformalPredictor
    >>> scp = SpatialConformalPredictor(
    ...     model=fitted_lgbm,
    ...     nonconformity='pearson_tweedie',
    ...     tweedie_power=1.5,
    ...     bandwidth_km=20.0,
    ... )
    >>> scp.calibrate(X_cal, y_cal, lat=lat_cal, lon=lon_cal)
    >>> result = scp.predict_interval(X_test, lat=lat_test, lon=lon_test, alpha=0.10)
    >>> print(result.lower[:5], result.upper[:5])

    Using postcodes instead of coordinates:

    >>> from insurance_spatial.conformal import PostcodeGeocoder
    >>> gc = PostcodeGeocoder()
    >>> lat_cal, lon_cal = gc.geocode(postcodes_cal)
    >>> scp.calibrate(X_cal, y_cal, lat=lat_cal, lon=lon_cal)
    """

    def __init__(
        self,
        model: Any,
        nonconformity: str = "pearson_tweedie",
        tweedie_power: float = 1.5,
        spatial_kernel: str = "gaussian",
        bandwidth_km: Optional[float] = None,
        spread_model: Optional[Any] = None,
        n_eff_min: int = 30,
    ) -> None:
        if not hasattr(model, "predict"):
            raise TypeError("model must implement a predict() method")

        _valid_kernels = {"gaussian", "epanechnikov", "uniform"}
        if spatial_kernel not in _valid_kernels:
            raise ValueError(
                f"spatial_kernel must be one of {sorted(_valid_kernels)}, "
                f"got '{spatial_kernel}'"
            )

        if bandwidth_km is not None and float(bandwidth_km) <= 0:
            raise ValueError(f"bandwidth_km must be positive, got {bandwidth_km}")

        self.model = model
        self.nonconformity = nonconformity
        self.tweedie_power = float(tweedie_power)
        self.spatial_kernel = spatial_kernel
        self.bandwidth_km = bandwidth_km
        self.spread_model = spread_model
        self.n_eff_min = int(n_eff_min)

        # Set after calibrate()
        self.cal_scores_: Optional[np.ndarray] = None
        self.cal_lat_: Optional[np.ndarray] = None
        self.cal_lon_: Optional[np.ndarray] = None
        self.cal_yhat_: Optional[np.ndarray] = None
        self.bandwidth_km_: Optional[float] = None
        self.is_calibrated_: bool = False
        self._score_fn = make_score(nonconformity, tweedie_power, spread_model)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        X_cal: Any,
        y_cal: np.ndarray,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        postcodes: Optional[list[str]] = None,
        exposure: Optional[np.ndarray] = None,
        cv_candidates_km: Optional[list[float]] = None,
        cv_folds: int = 5,
    ) -> CalibrationResult:
        """
        Compute non-conformity scores on the calibration set.

        Must be called before predict_interval(). Provide either (lat, lon) arrays
        or postcodes; if postcodes are provided, they are geocoded automatically.

        Parameters
        ----------
        X_cal : array-like, shape (n_cal, p)
            Feature matrix for the calibration set.
        y_cal : array-like, shape (n_cal,)
            Observed outcomes (losses, claim counts, etc.).
        lat : np.ndarray, optional
            Calibration point latitudes. Must provide either this or postcodes.
        lon : np.ndarray, optional
            Calibration point longitudes.
        postcodes : list of str, optional
            UK postcodes. Geocoded automatically if provided.
        exposure : np.ndarray, optional
            Exposure values, shape (n_cal,). If provided, model predictions are
            multiplied by exposure before computing scores.
        cv_candidates_km : list of float, optional
            Candidate bandwidths for CV selection. Only used when bandwidth_km=None.
            Defaults to [2, 5, 10, 15, 20, 30, 50].
        cv_folds : int, default 5
            Number of spatial CV folds for bandwidth selection.

        Returns
        -------
        CalibrationResult
            Summary statistics of the calibration.

        Raises
        ------
        ValueError
            If neither lat/lon nor postcodes are provided.
        """
        lat, lon = self._resolve_coords(lat, lon, postcodes, label="calibration")
        y_cal = np.asarray(y_cal, dtype=float)

        # Get model predictions
        yhat = self._predict_model(X_cal, exposure)

        # Compute non-conformity scores
        if self.nonconformity == "scaled_absolute":
            scores = self._score_fn.score(y_cal, yhat, X=X_cal)
        else:
            scores = self._score_fn.score(y_cal, yhat)

        # Resolve bandwidth
        if self.bandwidth_km is not None:
            bw = float(self.bandwidth_km)
            bw_from_cv = False
        else:
            from insurance_spatial.conformal.bandwidth import BandwidthSelector
            selector = BandwidthSelector(
                candidates_km=cv_candidates_km or [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
                cv=cv_folds,
            )
            cv_result = selector.select(scores, lat, lon, alpha=0.10)
            bw = cv_result.optimal_km
            bw_from_cv = True

        self.cal_scores_ = scores
        self.cal_lat_ = lat
        self.cal_lon_ = lon
        self.cal_yhat_ = yhat
        self.bandwidth_km_ = bw
        self.is_calibrated_ = True

        return CalibrationResult(
            n_calibration=len(scores),
            bandwidth_km=bw,
            bandwidth_selected_by_cv=bw_from_cv,
            score_name=self._score_fn.name,
            score_mean=float(np.mean(scores)),
            score_std=float(np.std(scores)),
            score_p50=float(np.percentile(scores, 50)),
            score_p95=float(np.percentile(scores, 95)),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_interval(
        self,
        X_test: Any,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        postcodes: Optional[list[str]] = None,
        alpha: float = 0.10,
        exposure: Optional[np.ndarray] = None,
        bandwidth_km: Optional[float] = None,
    ) -> IntervalResult:
        """
        Produce spatially weighted prediction intervals for test points.

        For each test point, the conformal threshold is computed as the weighted
        (1-alpha) quantile of calibration non-conformity scores, with weights
        determined by geographic proximity (Gaussian kernel).

        The finite-sample validity adjustment (Tibshirani 2019) adds an augmented
        point at +infinity with weight proportional to 1/(n_cal + 1) to ensure
        the marginal coverage guarantee holds exactly at 1 - alpha.

        Parameters
        ----------
        X_test : array-like, shape (n_test, p)
            Feature matrix for test points.
        lat : np.ndarray, optional
            Test point latitudes. Provide either this or postcodes.
        lon : np.ndarray, optional
            Test point longitudes.
        postcodes : list of str, optional
            UK postcodes for test points.
        alpha : float, default 0.10
            Miscoverage level. alpha=0.10 gives 90% coverage intervals.
        exposure : np.ndarray, optional
            Test exposure values, shape (n_test,).
        bandwidth_km : float, optional
            Override the calibrated bandwidth for this prediction only.

        Returns
        -------
        IntervalResult
            Contains lower, upper, point, alpha, n_effective, bandwidth_km.

        Raises
        ------
        RuntimeError
            If calibrate() has not been called.

        Examples
        --------
        >>> result = scp.predict_interval(X_test, lat=lat_test, lon=lon_test)
        >>> covered = (y_test >= result.lower) & (y_test <= result.upper)
        >>> print(f"Coverage: {covered.mean():.3f}")
        """
        self._check_calibrated()

        lat_test, lon_test = self._resolve_coords(lat, lon, postcodes, label="test")
        yhat_test = self._predict_model(X_test, exposure)

        bw = float(bandwidth_km) if bandwidth_km is not None else self.bandwidth_km_

        # Compute weights: shape (n_cal, n_test)
        W = compute_weights(
            self.cal_lat_, self.cal_lon_,
            lat_test, lon_test,
            bw, self.spatial_kernel,
        )

        n_cal = len(self.cal_scores_)
        n_test = len(yhat_test)
        lower = np.empty(n_test)
        upper = np.empty(n_test)
        n_eff_arr = np.empty(n_test)

        for j in range(n_test):
            w_j = W[:, j]
            n_eff_arr[j] = kish_n_eff(w_j)

            q = self._weighted_quantile(self.cal_scores_, w_j, alpha)

            if self.nonconformity == "scaled_absolute":
                # Need X for spread model
                X_j = X_test[j : j + 1] if hasattr(X_test, "__getitem__") else X_test
                lo, hi = self._score_fn.invert(yhat_test[j : j + 1], q, X=X_j)
                lower[j] = float(lo[0])
                upper[j] = float(hi[0])
            else:
                lo, hi = self._score_fn.invert(yhat_test[j : j + 1], q)
                lower[j] = float(lo[0])
                upper[j] = float(hi[0])

        # Warn about low effective N
        low_n_eff = np.sum(n_eff_arr < self.n_eff_min)
        if low_n_eff > 0:
            warnings.warn(
                f"{low_n_eff} test point(s) have effective calibration N < {self.n_eff_min} "
                f"(min: {n_eff_arr.min():.1f}). "
                "Intervals at these points may be unreliable. "
                "Consider increasing bandwidth_km or collecting more calibration data in the area.",
                UserWarning,
                stacklevel=2,
            )

        return IntervalResult(
            lower=lower,
            upper=upper,
            point=yhat_test,
            alpha=alpha,
            n_effective=n_eff_arr,
            bandwidth_km=bw,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def spatial_coverage_report(
        self,
        X_val: Any,
        y_val: np.ndarray,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        postcodes: Optional[list[str]] = None,
        alpha: float = 0.10,
    ) -> "SpatialCoverageReport":
        """
        Return a SpatialCoverageReport for a labelled validation set.

        Convenience method — equivalent to:
            SpatialCoverageReport(self).evaluate(X_val, y_val, ...)

        Parameters
        ----------
        X_val : array-like
        y_val : array-like
        lat, lon : np.ndarray, optional
        postcodes : list of str, optional
        alpha : float, default 0.10

        Returns
        -------
        SpatialCoverageReport (not yet evaluated; call .evaluate() or use the
        CoverageResult from evaluate())
        """
        from insurance_spatial.conformal.report import SpatialCoverageReport

        report = SpatialCoverageReport(self)
        report.evaluate(X_val, y_val, lat=lat, lon=lon, postcodes=postcodes, alpha=alpha)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_calibrated(self) -> None:
        if not self.is_calibrated_:
            raise RuntimeError(
                "Predictor has not been calibrated. Call calibrate() first."
            )

    def _resolve_coords(
        self,
        lat: Optional[np.ndarray],
        lon: Optional[np.ndarray],
        postcodes: Optional[list[str]],
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve coordinates from explicit lat/lon or postcodes."""
        if lat is not None and lon is not None:
            return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        elif postcodes is not None:
            from insurance_spatial.conformal.geocoder import PostcodeGeocoder
            gc = PostcodeGeocoder()
            return gc.geocode(postcodes)
        else:
            raise ValueError(
                f"Must provide either (lat, lon) arrays or postcodes for {label} points."
            )

    def _predict_model(
        self, X: Any, exposure: Optional[np.ndarray]
    ) -> np.ndarray:
        """Run model.predict() and apply exposure if provided."""
        yhat = np.asarray(self.model.predict(X), dtype=float)
        if exposure is not None:
            exp = np.asarray(exposure, dtype=float)
            if exp.shape != yhat.shape:
                raise ValueError(
                    f"exposure shape {exp.shape} does not match predictions {yhat.shape}"
                )
            yhat = yhat * exp
        return yhat

    def _weighted_quantile(
        self,
        scores: np.ndarray,
        weights: np.ndarray,
        alpha: float,
    ) -> float:
        """
        Weighted (1-alpha) quantile with Tibshirani (2019) augmentation.

        Adds an augmented point at +infinity representing a hypothetical
        calibration point located at the test position.  Under the Gaussian
        kernel, a point at distance zero from itself has weight exp(0) = 1.0,
        so the augmented weight is 1.0.

        This is the correct implementation per Tibshirani (2019) Theorem 1 for
        spatially weighted conformal prediction.  The previous implementation
        used sum(w)/n as the augmented weight, which approximates the average
        calibration weight and is wrong when calibration weights are non-uniform
        (i.e., almost always in the spatial setting).  For urban test points with
        many nearby calibration points, sum(w)/n << 1.0, making the augmented
        infinity point too light and under-protecting marginal coverage.

        Note: bandwidth.py's _weighted_quantile already uses weight=1.0 (correct).
        This method is now consistent with that implementation.
        """
        n = len(scores)
        w = np.asarray(weights, dtype=float)

        # P0 fix: augmented infinity point weight = 1.0, matching kernel(0) = exp(0) = 1
        # for Gaussian kernel.  bandwidth.py correctly uses np.append(weights, 1.0);
        # this was inconsistently using sum(w)/n, which is the average weight and wrong
        # for non-uniform spatial weights.
        w_aug = np.append(w, 1.0)
        s_aug = np.append(scores, np.inf)

        w_total = np.sum(w_aug)
        if w_total < 1e-16:
            # Degenerate: no weight — fall back to uniform quantile
            return float(np.quantile(scores, 1.0 - alpha))

        w_norm = w_aug / w_total

        sort_idx = np.argsort(s_aug)
        s_sorted = s_aug[sort_idx]
        w_sorted = w_norm[sort_idx]
        cdf = np.cumsum(w_sorted)

        idx = np.searchsorted(cdf, 1.0 - alpha)
        if idx >= len(s_sorted):
            return float(s_sorted[-1])
        return float(s_sorted[idx])
