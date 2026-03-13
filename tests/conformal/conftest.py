"""
Pytest fixtures for insurance-spatial-conformal tests.

All synthetic data is generated here so individual test modules stay focused
on assertions rather than setup. The data mimics a UK home insurance portfolio:
coordinates clustered around real UK cities, Tweedie-distributed losses.
"""

from __future__ import annotations

import numpy as np
import pytest


# ------------------------------------------------------------------
# Random state
# ------------------------------------------------------------------

RNG_SEED = 42


# ------------------------------------------------------------------
# Dummy model
# ------------------------------------------------------------------

class DummyModel:
    """A trivial predict-able model for testing without ML dependencies."""

    def __init__(self, scale: float = 500.0, rng_seed: int = 0) -> None:
        self.scale = scale
        self._rng = np.random.default_rng(rng_seed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        # Predictions roughly proportional to first feature, with noise
        base = np.abs(X[:, 0]) * self.scale if hasattr(X, "__getitem__") else np.array([self.scale])
        return np.clip(base + 50.0, 10.0, None)


class ConstantModel:
    """Always predicts the same value. Useful for isolating score behaviour."""

    def __init__(self, value: float = 200.0) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.value)


class SpreadModel:
    """Dummy spread model predicting constant sigma."""

    def __init__(self, sigma: float = 100.0) -> None:
        self.sigma = sigma

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.sigma)


# ------------------------------------------------------------------
# Coordinate fixtures
# ------------------------------------------------------------------

def _uk_city_coords(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic UK-like coordinates clustered around cities.

    Generates points near London, Manchester, Birmingham, Glasgow, Leeds.
    """
    centres = [
        (51.51, -0.13),   # London
        (53.48, -2.24),   # Manchester
        (52.49, -1.90),   # Birmingham
        (55.86, -4.25),   # Glasgow
        (53.80, -1.55),   # Leeds
    ]
    n_per_city = n // len(centres)
    lats, lons = [], []
    for lat_c, lon_c in centres:
        k = n_per_city + (1 if len(lats) < n % len(centres) else 0)
        lats.append(rng.normal(lat_c, 0.3, size=k))
        lons.append(rng.normal(lon_c, 0.4, size=k))
    lat = np.concatenate(lats)[:n]
    lon = np.concatenate(lons)[:n]
    return lat, lon


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def cal_data(rng):
    """Calibration set: 400 observations, UK-like coordinates."""
    n = 400
    lat, lon = _uk_city_coords(n, rng)
    X = rng.normal(size=(n, 5))
    X[:, 0] = np.clip(X[:, 0], 0.1, None)  # first feature positive for DummyModel
    yhat = np.clip(np.abs(X[:, 0]) * 500.0 + 50.0, 10.0, None)
    noise = rng.gamma(shape=2.0, scale=1.0, size=n)
    y = yhat * noise  # rough Tweedie-like
    return {"X": X, "y": y, "lat": lat, "lon": lon, "yhat": yhat}


@pytest.fixture(scope="session")
def test_data(rng):
    """Test set: 100 observations, UK-like coordinates."""
    n = 100
    lat, lon = _uk_city_coords(n, rng)
    X = rng.normal(size=(n, 5))
    X[:, 0] = np.clip(X[:, 0], 0.1, None)
    yhat = np.clip(np.abs(X[:, 0]) * 500.0 + 50.0, 10.0, None)
    noise = rng.gamma(shape=2.0, scale=1.0, size=n)
    y = yhat * noise
    return {"X": X, "y": y, "lat": lat, "lon": lon, "yhat": yhat}


@pytest.fixture(scope="session")
def val_data(rng):
    """Validation set: 200 observations."""
    n = 200
    lat, lon = _uk_city_coords(n, rng)
    X = rng.normal(size=(n, 5))
    X[:, 0] = np.clip(X[:, 0], 0.1, None)
    yhat = np.clip(np.abs(X[:, 0]) * 500.0 + 50.0, 10.0, None)
    noise = rng.gamma(shape=2.0, scale=1.0, size=n)
    y = yhat * noise
    return {"X": X, "y": y, "lat": lat, "lon": lon, "yhat": yhat}


@pytest.fixture(scope="session")
def dummy_model() -> DummyModel:
    return DummyModel()


@pytest.fixture(scope="session")
def constant_model() -> ConstantModel:
    return ConstantModel(200.0)


@pytest.fixture(scope="session")
def spread_model() -> SpreadModel:
    return SpreadModel(100.0)


@pytest.fixture(scope="session")
def calibrated_predictor(dummy_model, cal_data):
    """A pre-calibrated SpatialConformalPredictor with a fixed 20 km bandwidth."""
    from insurance_spatial.conformal import SpatialConformalPredictor

    scp = SpatialConformalPredictor(
        model=dummy_model,
        nonconformity="pearson_tweedie",
        tweedie_power=1.5,
        spatial_kernel="gaussian",
        bandwidth_km=20.0,
    )
    scp.calibrate(
        cal_data["X"],
        cal_data["y"],
        lat=cal_data["lat"],
        lon=cal_data["lon"],
    )
    return scp
