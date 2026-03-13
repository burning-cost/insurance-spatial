"""
Type definitions for insurance-spatial-conformal.

Centralises type aliases and dataclasses so the rest of the library imports from
one place rather than having type definitions scattered across modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ------------------------------------------------------------------
# Coordinate arrays
# ------------------------------------------------------------------

Coords = tuple[np.ndarray, np.ndarray]
"""(lat, lon) pair, both shape (n,), decimal degrees WGS-84."""


# ------------------------------------------------------------------
# Literal type aliases
# ------------------------------------------------------------------

KernelType = Literal["gaussian", "epanechnikov", "uniform"]
"""Supported spatial kernel families."""

NonconformityType = Literal[
    "pearson_tweedie",
    "pearson",
    "absolute",
    "scaled_absolute",
    "deviance",
]
"""Supported non-conformity score types."""

CVMetric = Literal["macg", "interval_width"]
"""Cross-validation objective for bandwidth selection."""


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Outcome of SpatialConformalPredictor.calibrate()."""

    n_calibration: int
    """Number of calibration observations."""

    bandwidth_km: float
    """Gaussian kernel bandwidth in kilometres, either supplied or CV-selected."""

    bandwidth_selected_by_cv: bool
    """True when bandwidth was determined by cross-validation rather than supplied."""

    score_name: str
    """Non-conformity score name used."""

    score_mean: float
    """Mean non-conformity score across calibration set."""

    score_std: float
    """Standard deviation of non-conformity scores."""

    score_p50: float
    """Median non-conformity score."""

    score_p95: float
    """95th percentile of non-conformity score — rough indicator of tail behaviour."""


@dataclass
class IntervalResult:
    """Prediction intervals from SpatialConformalPredictor.predict_interval()."""

    lower: np.ndarray
    """Lower bound of prediction interval, shape (n_test,)."""

    upper: np.ndarray
    """Upper bound of prediction interval, shape (n_test,)."""

    point: np.ndarray
    """Point prediction from the wrapped model, shape (n_test,)."""

    alpha: float
    """Miscoverage level used."""

    n_effective: np.ndarray
    """Effective sample size (Kish formula) at each test point, shape (n_test,)."""

    bandwidth_km: float
    """Bandwidth used for this prediction."""

    def width(self) -> np.ndarray:
        """Interval width = upper - lower."""
        return self.upper - self.lower

    def relative_width(self) -> np.ndarray:
        """Interval width relative to point prediction. Useful for comparing across risk levels."""
        denom = np.where(self.point > 0, self.point, np.nan)
        return self.width() / denom


@dataclass
class CoverageResult:
    """Output of SpatialCoverageReport.evaluate()."""

    alpha: float
    """Target miscoverage level."""

    target_coverage: float
    """Nominal coverage target = 1 - alpha."""

    marginal_coverage: float
    """Overall empirical coverage across the validation set."""

    macg: float
    """Mean Absolute Coverage Gap across spatial grid cells."""

    n_grid_cells: int
    """Number of grid cells used for MACG calculation."""

    n_val: int
    """Number of validation observations."""

    coverage_by_cell: Optional[np.ndarray] = None
    """Per-cell empirical coverage, shape (n_grid_cells,). None if not computed."""

    cell_centres_lat: Optional[np.ndarray] = None
    """Grid cell centre latitudes."""

    cell_centres_lon: Optional[np.ndarray] = None
    """Grid cell centre longitudes."""

    coverage_by_region: Optional[dict] = None
    """Coverage broken down by named region/segment."""

    def coverage_gap(self) -> float:
        """Signed gap: target - marginal. Positive = under-covered."""
        return self.target_coverage - self.marginal_coverage


@dataclass
class BandwidthCVResult:
    """Result of BandwidthSelector.select()."""

    optimal_km: float
    """Optimal bandwidth in km."""

    candidates_km: list[float]
    """Candidate bandwidths evaluated."""

    cv_scores: list[float]
    """Mean CV score for each candidate (lower = better)."""

    metric: str
    """Metric optimised (e.g. 'macg')."""

    n_folds: int
    """Number of spatial CV folds used."""
