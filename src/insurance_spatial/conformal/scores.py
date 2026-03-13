"""
Non-conformity scores for spatially weighted conformal prediction.

The non-conformity score is the core design choice. For insurance pricing, the
Tweedie Pearson score is the right default: it divides the absolute residual by
yhat^(p/2), which is the standard deviation under a Tweedie(p) assumption. This
variance-stabilises the scores, so the spatial kernel weighting isn't confounded
by the model's own heteroscedasticity.

The score hierarchy (tighter intervals, same coverage):
    pearson_tweedie > pearson > scaled_absolute > absolute

References:
    Hjort, Jullum, Loland (2025), arXiv:2312.06531
    Manna et al. (2025), arXiv:2507.06921
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def _validate(
    y: np.ndarray, yhat: np.ndarray, clip: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shapes and clip predictions away from zero."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError(
            f"y and yhat must have the same shape: got {y.shape} vs {yhat.shape}"
        )
    if np.any(yhat <= 0):
        n_bad = int(np.sum(yhat <= 0))
        warnings.warn(
            f"{n_bad} predictions are <= 0 and have been clipped to {clip}. "
            "Non-positive predictions are invalid for Tweedie/Poisson families.",
            UserWarning,
            stacklevel=3,
        )
        yhat = np.clip(yhat, clip, None)
    return y, yhat


class TweediePearsonScore:
    """
    Tweedie Pearson non-conformity score: |y - yhat| / yhat^(p/2).

    Under a Tweedie(p) distribution Var(Y) proportional to mu^p, so yhat^(p/2) is the
    expected standard deviation. Dividing by this quantity variance-stabilises the
    score and produces tighter intervals than the raw absolute residual.

    This is the recommended default for combined ratio / severity / burning cost
    models built with Tweedie GLM or LightGBM/XGBoost with tweedie objective.

    Parameters
    ----------
    power : float, default 1.5
        Tweedie variance power. Common choices:
        - 0: normal
        - 1: Poisson (frequency)
        - 1.5: compound Poisson-Gamma (burning cost, the usual default)
        - 2: Gamma (pure severity)
        - 3: inverse Gaussian (heavy tail severity)

    Examples
    --------
    >>> score_fn = TweediePearsonScore(power=1.5)
    >>> scores = score_fn.score(y_cal, yhat_cal)
    >>> y_lo, y_hi = score_fn.invert(yhat_test, threshold=q)
    """

    def __init__(self, power: float = 1.5) -> None:
        if not isinstance(power, (int, float)):
            raise TypeError(f"power must be numeric, got {type(power)}")
        self.power = float(power)
        self.name = "pearson_tweedie"

    def score(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Compute non-conformity scores.

        Parameters
        ----------
        y : array-like, shape (n,)
            Observed values (non-negative for insurance losses).
        yhat : array-like, shape (n,)
            Model point predictions (must be positive).

        Returns
        -------
        np.ndarray, shape (n,)
            Non-conformity scores, all >= 0.
        """
        y, yhat = _validate(y, yhat)
        denom = yhat ** (self.power / 2.0)
        return np.abs(y - yhat) / denom

    def invert(self, yhat: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a score threshold back to prediction interval bounds.

        Solves |y - yhat| / yhat^(p/2) <= q  =>  y in [yhat - q * yhat^(p/2),
        yhat + q * yhat^(p/2)].

        Parameters
        ----------
        yhat : array-like, shape (n,)
            Point predictions.
        threshold : float
            Score quantile (the conformal threshold).

        Returns
        -------
        lower, upper : np.ndarray, np.ndarray
            Prediction interval bounds, both shape (n,).
        """
        yhat = np.asarray(yhat, dtype=float)
        denom = yhat ** (self.power / 2.0)
        margin = threshold * denom
        lower = np.maximum(yhat - margin, 0.0)
        upper = yhat + margin
        return lower, upper


class AbsoluteScore:
    """
    Absolute residual: |y - yhat|.

    The baseline score. Not recommended for insurance data — ignores
    heteroscedasticity completely, so high-risk policies get the same absolute
    tolerance as low-risk ones. Use only as a diagnostic or when comparing against
    naive methods.

    Parameters
    ----------
    None

    Examples
    --------
    >>> score_fn = AbsoluteScore()
    >>> scores = score_fn.score(y_cal, yhat_cal)
    """

    def __init__(self) -> None:
        self.name = "absolute"

    def score(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        y : array-like
        yhat : array-like

        Returns
        -------
        np.ndarray
        """
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(yhat, dtype=float)
        if y.shape != yhat.shape:
            raise ValueError(f"Shape mismatch: {y.shape} vs {yhat.shape}")
        return np.abs(y - yhat)

    def invert(self, yhat: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        yhat : array-like
        threshold : float

        Returns
        -------
        lower, upper : np.ndarray, np.ndarray
        """
        yhat = np.asarray(yhat, dtype=float)
        lower = np.maximum(yhat - threshold, 0.0)
        upper = yhat + threshold
        return lower, upper


class ScaledAbsoluteScore:
    """
    Scaled absolute residual: |y - yhat| / spread(X).

    A two-model approach: the primary model gives yhat; a secondary spread model
    sigma(X) estimates local residual scale. The score |y - yhat| / sigma(X)
    normalises by predicted difficulty rather than predicted level.

    This is Hjort et al.'s 'Normalized-2' score — typically the tightest intervals
    when the spread model is accurate.

    Parameters
    ----------
    spread_model : fitted model
        Any model with a predict(X) method returning predicted spread (sigma).
        Train on |y - yhat| from your primary model.

    Examples
    --------
    >>> spread_model = LGBMRegressor().fit(X_cal, np.abs(y_cal - yhat_cal))
    >>> score_fn = ScaledAbsoluteScore(spread_model)
    >>> scores = score_fn.score(y_cal, yhat_cal, X=X_cal)
    """

    def __init__(self, spread_model: object) -> None:
        if not hasattr(spread_model, "predict"):
            raise TypeError("spread_model must have a predict() method")
        self.spread_model = spread_model
        self.name = "scaled_absolute"

    def score(
        self,
        y: np.ndarray,
        yhat: np.ndarray,
        X: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        y : array-like
        yhat : array-like
        X : array-like, optional
            Features passed to spread_model.predict(). Required if sigma is None.
        sigma : array-like, optional
            Pre-computed spread predictions. If provided, X is ignored.

        Returns
        -------
        np.ndarray
        """
        y, yhat = _validate(y, yhat)
        if sigma is None:
            if X is None:
                raise ValueError("Either X or sigma must be provided for ScaledAbsoluteScore")
            sigma = np.asarray(self.spread_model.predict(X), dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.clip(sigma, 1e-8, None)
        return np.abs(y - yhat) / sigma

    def invert(
        self,
        yhat: np.ndarray,
        threshold: float,
        X: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        yhat : array-like
        threshold : float
        X : array-like, optional
        sigma : array-like, optional

        Returns
        -------
        lower, upper : np.ndarray, np.ndarray
        """
        yhat = np.asarray(yhat, dtype=float)
        if sigma is None:
            if X is None:
                raise ValueError("Either X or sigma must be provided")
            sigma = np.asarray(self.spread_model.predict(X), dtype=float)
        sigma = np.clip(np.asarray(sigma, dtype=float), 1e-8, None)
        margin = threshold * sigma
        lower = np.maximum(yhat - margin, 0.0)
        upper = yhat + margin
        return lower, upper


class PearsonScore:
    """
    Standard Pearson residual: |y - yhat| / sqrt(yhat).

    Appropriate for Poisson data (p=1). Equivalent to TweediePearsonScore(power=1)
    but kept as a named class for clarity in frequency modelling contexts.

    Parameters
    ----------
    None

    Examples
    --------
    >>> score_fn = PearsonScore()
    >>> scores = score_fn.score(y_cal, yhat_cal)
    """

    def __init__(self) -> None:
        self.name = "pearson"

    def score(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        y : array-like
        yhat : array-like

        Returns
        -------
        np.ndarray
        """
        y, yhat = _validate(y, yhat)
        return np.abs(y - yhat) / np.sqrt(yhat)

    def invert(self, yhat: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        yhat : array-like
        threshold : float

        Returns
        -------
        lower, upper : np.ndarray, np.ndarray
        """
        yhat = np.asarray(yhat, dtype=float)
        margin = threshold * np.sqrt(yhat)
        lower = np.maximum(yhat - margin, 0.0)
        upper = yhat + margin
        return lower, upper


def make_score(
    nonconformity: str,
    tweedie_power: float = 1.5,
    spread_model: Optional[object] = None,
) -> TweediePearsonScore | AbsoluteScore | ScaledAbsoluteScore | PearsonScore:
    """
    Factory function for non-conformity score objects.

    Parameters
    ----------
    nonconformity : str
        One of 'pearson_tweedie', 'pearson', 'absolute', 'scaled_absolute'.
    tweedie_power : float, default 1.5
        Used only for 'pearson_tweedie'.
    spread_model : fitted model, optional
        Required for 'scaled_absolute'.

    Returns
    -------
    Score object with .score() and .invert() methods.

    Raises
    ------
    ValueError
        If nonconformity string is not recognised.
    """
    mapping = {
        "pearson_tweedie": lambda: TweediePearsonScore(tweedie_power),
        "pearson": lambda: PearsonScore(),
        "absolute": lambda: AbsoluteScore(),
        "scaled_absolute": lambda: ScaledAbsoluteScore(spread_model),
    }
    if nonconformity not in mapping:
        raise ValueError(
            f"Unknown nonconformity score '{nonconformity}'. "
            f"Choose from: {sorted(mapping.keys())}"
        )
    if nonconformity == "scaled_absolute" and spread_model is None:
        raise ValueError(
            "spread_model is required for 'scaled_absolute' score. "
            "Pass a fitted model that predicts |y - yhat|."
        )
    return mapping[nonconformity]()
