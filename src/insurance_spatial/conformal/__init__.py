"""
insurance_spatial.conformal: spatially weighted conformal prediction for insurance pricing.

Standard conformal prediction gives nationally correct coverage but can be badly
miscalibrated geographically. This sub-package fixes that using Gaussian kernel
weighting of calibration non-conformity scores, so that prediction intervals for
a test property in Taunton are informed primarily by similar calibration properties
in Somerset, not by claims from inner London.

Based on the spatially weighted conformal prediction framework of Hjort, Jullum, and
Loland (2025), adapted for UK insurance pricing with Tweedie Pearson non-conformity
scores and cross-validated bandwidth selection.

Quickstart
----------
>>> from insurance_spatial.conformal import SpatialConformalPredictor
>>> scp = SpatialConformalPredictor(
...     model=fitted_model,
...     nonconformity='pearson_tweedie',
...     tweedie_power=1.5,
... )
>>> scp.calibrate(X_cal, y_cal, lat=lat_cal, lon=lon_cal)
>>> result = scp.predict_interval(X_test, lat=lat_test, lon=lon_test, alpha=0.10)
>>> print(result.lower, result.upper)

References
----------
Hjort, N. L., Jullum, M., & Loland, A. (2025). Uncertainty quantification in
automated valuation models with spatially weighted conformal prediction.
arXiv:2312.06531 / IJDSA (Springer). doi:10.1007/s41060-025-00862-4

Tibshirani, R. J., Barber, R. F., Candes, E. J., & Ramdas, A. (2019). Conformal
prediction under covariate shift. NeurIPS 2019.
"""

from insurance_spatial.conformal.predictor import SpatialConformalPredictor
from insurance_spatial.conformal.scores import (
    TweediePearsonScore,
    AbsoluteScore,
    PearsonScore,
    ScaledAbsoluteScore,
    make_score,
)
from insurance_spatial.conformal.bandwidth import BandwidthSelector
from insurance_spatial.conformal.report import SpatialCoverageReport
from insurance_spatial.conformal.geocoder import PostcodeGeocoder
from insurance_spatial.conformal._kernel import (
    haversine_distances,
    gaussian_weights,
    kish_n_eff,
)
from insurance_spatial.conformal._types import (
    CalibrationResult,
    IntervalResult,
    CoverageResult,
    BandwidthCVResult,
)

__all__ = [
    # Main predictor
    "SpatialConformalPredictor",
    # Score functions
    "TweediePearsonScore",
    "AbsoluteScore",
    "PearsonScore",
    "ScaledAbsoluteScore",
    "make_score",
    # Bandwidth selection
    "BandwidthSelector",
    # Diagnostics
    "SpatialCoverageReport",
    # Geocoding
    "PostcodeGeocoder",
    # Kernel utilities
    "haversine_distances",
    "gaussian_weights",
    "kish_n_eff",
    # Result types
    "CalibrationResult",
    "IntervalResult",
    "CoverageResult",
    "BandwidthCVResult",
]
