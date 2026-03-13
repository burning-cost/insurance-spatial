"""
Tests for _kernel.py: haversine distances and kernel weight functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_spatial.conformal._kernel import (
    haversine_distances,
    gaussian_weights,
    epanechnikov_weights,
    uniform_weights,
    compute_weights,
    kish_n_eff,
)


class TestHaversineDistances:
    def test_same_point_zero_distance(self):
        lat = np.array([51.5])
        lon = np.array([-0.1])
        d = haversine_distances(lat, lon, lat, lon)
        assert d.shape == (1, 1)
        assert float(d[0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_london_to_manchester_approx_260km(self):
        lat1 = np.array([51.51])
        lon1 = np.array([-0.13])
        lat2 = np.array([53.48])
        lon2 = np.array([-2.24])
        d = haversine_distances(lat1, lon1, lat2, lon2)
        assert 250.0 < float(d[0, 0]) < 275.0

    def test_output_shape(self):
        lat1 = np.array([51.5, 52.0, 53.0])
        lon1 = np.array([-0.1, -1.0, -2.0])
        lat2 = np.array([51.5, 55.0])
        lon2 = np.array([-0.1, -4.0])
        d = haversine_distances(lat1, lon1, lat2, lon2)
        assert d.shape == (3, 2)

    def test_symmetry(self):
        lat1 = np.array([51.5])
        lon1 = np.array([-0.1])
        lat2 = np.array([53.5])
        lon2 = np.array([-2.2])
        d12 = haversine_distances(lat1, lon1, lat2, lon2)
        d21 = haversine_distances(lat2, lon2, lat1, lon1)
        assert float(d12[0, 0]) == pytest.approx(float(d21[0, 0]), rel=1e-6)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        lat1 = rng.uniform(50, 58, 20)
        lon1 = rng.uniform(-5, 2, 20)
        lat2 = rng.uniform(50, 58, 10)
        lon2 = rng.uniform(-5, 2, 10)
        d = haversine_distances(lat1, lon1, lat2, lon2)
        assert np.all(d >= 0)

    def test_euclid_vs_haversine_error_at_55N(self):
        """Euclidean on lat/lon underestimates E-W distance at high latitude."""
        lat = np.array([55.0])
        lon1 = np.array([-3.0])
        lon2 = np.array([0.0])
        h = float(haversine_distances(lat, lon1, lat, lon2)[0, 0])
        # Naive Euclidean ignoring projection
        naive_euclid_km = np.sqrt((55.0 - 55.0) ** 2 + (0.0 - (-3.0)) ** 2) * 111.0
        # Haversine should be materially less — lon degree compression at 55N
        assert h < naive_euclid_km
        # Should be around 185-195 km (3 deg * ~63 km/deg at 55N)
        assert 180.0 < h < 200.0

    def test_glasgow_to_london_approx_550km(self):
        lat1 = np.array([55.86])
        lon1 = np.array([-4.25])
        lat2 = np.array([51.51])
        lon2 = np.array([-0.13])
        d = haversine_distances(lat1, lon1, lat2, lon2)
        assert 540.0 < float(d[0, 0]) < 580.0


class TestGaussianWeights:
    def test_weight_one_at_zero_distance(self):
        lat = np.array([51.5])
        lon = np.array([-0.1])
        w = gaussian_weights(lat, lon, lat, lon, bandwidth_km=10.0)
        assert float(w[0, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_weights_decay_with_distance(self):
        lat_c = np.array([51.5, 52.0, 53.0])
        lon_c = np.array([-0.1, -0.1, -0.1])
        lat_t = np.array([51.5])
        lon_t = np.array([-0.1])
        w = gaussian_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=10.0)
        assert float(w[0, 0]) > float(w[1, 0]) > float(w[2, 0])

    def test_output_shape(self):
        n_cal, n_test = 50, 10
        rng = np.random.default_rng(1)
        lat_c = rng.uniform(50, 56, n_cal)
        lon_c = rng.uniform(-4, 2, n_cal)
        lat_t = rng.uniform(50, 56, n_test)
        lon_t = rng.uniform(-4, 2, n_test)
        w = gaussian_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=20.0)
        assert w.shape == (n_cal, n_test)

    def test_weights_positive(self):
        rng = np.random.default_rng(2)
        lat_c = rng.uniform(50, 56, 30)
        lon_c = rng.uniform(-4, 2, 30)
        lat_t = rng.uniform(50, 56, 5)
        lon_t = rng.uniform(-4, 2, 5)
        w = gaussian_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=10.0)
        # Gaussian weights are non-negative; may underflow to 0.0 at large distances
        assert np.all(w >= 0)
        # At a 10 km bandwidth, at least some pairs within UK extent should be non-trivially weighted
        assert np.any(w > 1e-10)

    def test_invalid_bandwidth_raises(self):
        with pytest.raises(ValueError, match="positive"):
            gaussian_weights(
                np.array([51.5]), np.array([-0.1]),
                np.array([51.5]), np.array([-0.1]),
                bandwidth_km=-5.0,
            )

    def test_weight_halves_at_bandwidth_times_sqrt_ln2(self):
        """Weight should halve at d = bw * sqrt(ln(2)) by construction."""
        lat_cal = np.array([51.5])
        lon_cal = np.array([-0.1])
        bw = 100.0  # large bandwidth to work in nearly flat geometry

        from insurance_spatial.conformal._kernel import haversine_distances
        # Find a point ~bw * sqrt(ln(2)) away
        half_dist_km = bw * np.sqrt(np.log(2))
        # Move north: 1 degree lat ~ 111 km
        delta_lat = half_dist_km / 111.0
        lat_test = np.array([51.5 + delta_lat])
        lon_test = np.array([-0.1])

        w = gaussian_weights(lat_cal, lon_cal, lat_test, lon_test, bandwidth_km=bw)
        assert float(w[0, 0]) == pytest.approx(0.5, rel=0.02)


class TestEpanechnikovWeights:
    def test_zero_outside_bandwidth(self):
        lat_c = np.array([51.5])
        lon_c = np.array([-0.1])
        lat_t = np.array([53.5])  # ~220 km away
        lon_t = np.array([-0.1])
        w = epanechnikov_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=50.0)
        assert float(w[0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_one_at_zero_distance(self):
        lat = np.array([51.5])
        lon = np.array([-0.1])
        w = epanechnikov_weights(lat, lon, lat, lon, bandwidth_km=10.0)
        assert float(w[0, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        rng = np.random.default_rng(3)
        lat_c = rng.uniform(50, 56, 20)
        lon_c = rng.uniform(-4, 2, 20)
        lat_t = rng.uniform(50, 56, 5)
        lon_t = rng.uniform(-4, 2, 5)
        w = epanechnikov_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=20.0)
        assert np.all(w >= 0)


class TestUniformWeights:
    def test_binary_output(self):
        rng = np.random.default_rng(4)
        lat_c = rng.uniform(50, 56, 50)
        lon_c = rng.uniform(-4, 2, 50)
        lat_t = rng.uniform(50, 56, 10)
        lon_t = rng.uniform(-4, 2, 10)
        w = uniform_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=20.0)
        unique_vals = np.unique(w)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_same_point_is_one(self):
        lat = np.array([51.5])
        lon = np.array([-0.1])
        w = uniform_weights(lat, lon, lat, lon, bandwidth_km=5.0)
        assert float(w[0, 0]) == pytest.approx(1.0)


class TestComputeWeights:
    def test_dispatch_gaussian(self):
        lat_c = np.array([51.5])
        lon_c = np.array([-0.1])
        w1 = compute_weights(lat_c, lon_c, lat_c, lon_c, 10.0, kernel="gaussian")
        w2 = gaussian_weights(lat_c, lon_c, lat_c, lon_c, 10.0)
        np.testing.assert_array_almost_equal(w1, w2)

    def test_dispatch_epanechnikov(self):
        lat_c = np.array([51.5])
        lon_c = np.array([-0.1])
        w1 = compute_weights(lat_c, lon_c, lat_c, lon_c, 10.0, kernel="epanechnikov")
        w2 = epanechnikov_weights(lat_c, lon_c, lat_c, lon_c, 10.0)
        np.testing.assert_array_almost_equal(w1, w2)

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            compute_weights(
                np.array([51.5]), np.array([-0.1]),
                np.array([51.5]), np.array([-0.1]),
                10.0, kernel="square"
            )


class TestKishNEff:
    def test_uniform_weights_returns_n(self):
        n = 100
        w = np.ones(n)
        assert kish_n_eff(w) == pytest.approx(float(n), rel=1e-6)

    def test_single_point_returns_one(self):
        assert kish_n_eff(np.array([5.0])) == pytest.approx(1.0, rel=1e-6)

    def test_zero_weights_excluded(self):
        w = np.array([1.0, 1.0, 0.0, 0.0])
        assert kish_n_eff(w) == pytest.approx(2.0, rel=1e-6)

    def test_concentrated_weights_low_n_eff(self):
        """One dominant weight should give n_eff close to 1."""
        w = np.array([1000.0] + [0.001] * 99)
        assert kish_n_eff(w) < 5.0

    def test_empty_weights_returns_zero(self):
        assert kish_n_eff(np.array([])) == pytest.approx(0.0)

    def test_all_zero_weights(self):
        assert kish_n_eff(np.zeros(10)) == pytest.approx(0.0)
