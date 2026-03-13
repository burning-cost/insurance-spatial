"""
Spatial kernel functions and haversine distance.

Internal module — not part of the public API.

All distance calculations use haversine rather than Euclidean. At 55N latitude
(central Scotland), a degree of longitude is about 64 km vs 111 km for a degree
of latitude — roughly 42% difference. Euclidean on lat/lon coordinates would
produce elliptical kernels that are skewed north-south. Haversine is exact on a
sphere and correct to <0.5% in the UK.

The bandwidth parameter eta is in kilometres. Weights decay as:
    w = exp(-d_km^2 / eta^2)

where d_km is the haversine distance in kilometres and eta is the bandwidth in km.
This parameterisation means the weight halves at d = eta * sqrt(ln(2)) ≈ 0.832 * eta.
"""

from __future__ import annotations

import numpy as np

_EARTH_RADIUS_KM = 6371.0


def haversine_distances(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Haversine distances in kilometres between two sets of points.

    Computes the full pairwise distance matrix between the two point sets.

    Parameters
    ----------
    lat1 : np.ndarray, shape (m,)
        Latitudes of first point set, decimal degrees.
    lon1 : np.ndarray, shape (m,)
        Longitudes of first point set, decimal degrees.
    lat2 : np.ndarray, shape (n,)
        Latitudes of second point set.
    lon2 : np.ndarray, shape (n,)
        Longitudes of second point set.

    Returns
    -------
    np.ndarray, shape (m, n)
        Pairwise haversine distances in kilometres.

    Examples
    --------
    >>> d = haversine_distances(
    ...     np.array([51.5]), np.array([-0.1]),  # London
    ...     np.array([53.5]), np.array([-2.2]),  # Manchester
    ... )
    >>> round(float(d[0, 0]))  # approx 260 km
    261
    """
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    # Convert to radians, broadcasting to (m, n)
    lat1_r = np.deg2rad(lat1)[:, np.newaxis]  # (m, 1)
    lon1_r = np.deg2rad(lon1)[:, np.newaxis]
    lat2_r = np.deg2rad(lat2)[np.newaxis, :]  # (1, n)
    lon2_r = np.deg2rad(lon2)[np.newaxis, :]

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))

    return _EARTH_RADIUS_KM * c


def gaussian_weights(
    lat_cal: np.ndarray,
    lon_cal: np.ndarray,
    lat_test: np.ndarray,
    lon_test: np.ndarray,
    bandwidth_km: float,
) -> np.ndarray:
    """
    Gaussian kernel weights between calibration and test points.

    w_{i,j} = exp(-d(i, j)^2 / bandwidth_km^2)

    Note: we do NOT normalise here. Normalisation (dividing by the sum of
    weights including the augmentation term) is the predictor's responsibility,
    as it must also include the (n+1)-th term for finite-sample validity.

    Parameters
    ----------
    lat_cal : np.ndarray, shape (n_cal,)
    lon_cal : np.ndarray, shape (n_cal,)
    lat_test : np.ndarray, shape (n_test,)
    lon_test : np.ndarray, shape (n_test,)
    bandwidth_km : float
        Gaussian bandwidth in kilometres.

    Returns
    -------
    np.ndarray, shape (n_cal, n_test)
        Unnormalised kernel weights.

    Examples
    --------
    >>> w = gaussian_weights(
    ...     np.array([51.5, 51.6]), np.array([-0.1, -0.1]),
    ...     np.array([51.5]), np.array([-0.1]),
    ...     bandwidth_km=10.0,
    ... )
    >>> w.shape
    (2, 1)
    """
    if bandwidth_km <= 0:
        raise ValueError(f"bandwidth_km must be positive, got {bandwidth_km}")
    distances = haversine_distances(lat_cal, lon_cal, lat_test, lon_test)
    return np.exp(-(distances ** 2) / (bandwidth_km ** 2))


def epanechnikov_weights(
    lat_cal: np.ndarray,
    lon_cal: np.ndarray,
    lat_test: np.ndarray,
    lon_test: np.ndarray,
    bandwidth_km: float,
) -> np.ndarray:
    """
    Epanechnikov kernel weights.

    w_{i,j} = max(0, 1 - (d/bandwidth)^2)

    Compact support: zero weight beyond bandwidth_km. Slightly more efficient
    than Gaussian for large calibration sets, but requires enough observations
    within the bandwidth radius.

    Parameters
    ----------
    lat_cal, lon_cal, lat_test, lon_test : np.ndarray
    bandwidth_km : float

    Returns
    -------
    np.ndarray, shape (n_cal, n_test)
    """
    distances = haversine_distances(lat_cal, lon_cal, lat_test, lon_test)
    u = distances / bandwidth_km
    return np.maximum(0.0, 1.0 - u ** 2)


def uniform_weights(
    lat_cal: np.ndarray,
    lon_cal: np.ndarray,
    lat_test: np.ndarray,
    lon_test: np.ndarray,
    bandwidth_km: float,
) -> np.ndarray:
    """
    Uniform (nearest-neighbour) kernel weights.

    w_{i,j} = 1 if d(i, j) <= bandwidth_km, else 0.

    Equivalent to Hjort et al.'s NNCP method. Simple but can produce empty
    local calibration sets in sparse regions.

    Parameters
    ----------
    lat_cal, lon_cal, lat_test, lon_test : np.ndarray
    bandwidth_km : float

    Returns
    -------
    np.ndarray, shape (n_cal, n_test)
    """
    distances = haversine_distances(lat_cal, lon_cal, lat_test, lon_test)
    return (distances <= bandwidth_km).astype(float)


def compute_weights(
    lat_cal: np.ndarray,
    lon_cal: np.ndarray,
    lat_test: np.ndarray,
    lon_test: np.ndarray,
    bandwidth_km: float,
    kernel: str = "gaussian",
) -> np.ndarray:
    """
    Dispatch to the appropriate kernel function.

    Parameters
    ----------
    lat_cal, lon_cal : np.ndarray, shape (n_cal,)
    lat_test, lon_test : np.ndarray, shape (n_test,)
    bandwidth_km : float
    kernel : str, default 'gaussian'
        One of 'gaussian', 'epanechnikov', 'uniform'.

    Returns
    -------
    np.ndarray, shape (n_cal, n_test)

    Raises
    ------
    ValueError
        If kernel name is not recognised.
    """
    _kernels = {
        "gaussian": gaussian_weights,
        "epanechnikov": epanechnikov_weights,
        "uniform": uniform_weights,
    }
    if kernel not in _kernels:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Choose from: {sorted(_kernels.keys())}"
        )
    return _kernels[kernel](lat_cal, lon_cal, lat_test, lon_test, bandwidth_km)


def kish_n_eff(weights: np.ndarray) -> float:
    """
    Kish (1965) effective sample size for a weighted sample.

    n_eff = (sum w)^2 / sum(w^2)

    Measures how many equally-weighted observations the weighted sample is
    equivalent to. A good rule of thumb: if n_eff < 30, the local calibration
    is unreliable and the bandwidth should be widened.

    Parameters
    ----------
    weights : np.ndarray
        Non-negative weights. Need not be normalised.

    Returns
    -------
    float
        Effective sample size.
    """
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if len(w) == 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w ** 2))
