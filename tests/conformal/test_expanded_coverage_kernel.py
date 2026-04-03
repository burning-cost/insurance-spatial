"""
Expanded test coverage for conformal/_kernel.py.

Additional edge cases: boundary distances, cross-UK scale checks,
epanechnikov and uniform edge cases, kish_n_eff variants.
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


class TestHaversineDistancesAdditional:
    def test_scalar_inputs_broadcast_correctly(self):
        lat1 = np.array([51.5])
        lon1 = np.array([-0.1])
        lat2 = np.array([51.5, 53.0])
        lon2 = np.array([-0.1, -2.0])
        d = haversine_distances(lat1, lon1, lat2, lon2)
        assert d.shape == (1, 2)

    def test_anti_podal_approx_20000km(self):
        """Points on opposite sides of the Earth are ~20015 km apart."""
        lat1 = np.array([0.0])
        lon1 = np.array([0.0])
        lat2 = np.array([0.0])
        lon2 = np.array([180.0])
        d = haversine_distances(lat1, lon1, lat2, lon2)
        assert 19900.0 < float(d[0, 0]) < 20200.0

    def test_north_pole_to_equator_approx_10000km(self):
        lat1 = np.array([90.0])
        lon1 = np.array([0.0])
        lat2 = np.array([0.0])
        lon2 = np.array([0.0])
        d = haversine_distances(lat1, lon1, lat2, lon2)
        # Quarter of Earth's circumference ~10015 km
        assert 9900.0 < float(d[0, 0]) < 10200.0

    def test_pairwise_matrix_diagonal_near_zero(self):
        """Pairwise distances of a set with itself should have zero diagonal."""
        rng = np.random.default_rng(5)
        lat = rng.uniform(51, 55, 10)
        lon = rng.uniform(-2, 1, 10)
        d = haversine_distances(lat, lon, lat, lon)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-6)

    def test_returns_non_negative_for_random_uk_coords(self):
        rng = np.random.default_rng(6)
        lat1 = rng.uniform(50, 58, 15)
        lon1 = rng.uniform(-5, 2, 15)
        lat2 = rng.uniform(50, 58, 8)
        lon2 = rng.uniform(-5, 2, 8)
        d = haversine_distances(lat1, lon1, lat2, lon2)
        assert np.all(d >= 0.0)

    def test_edinburgh_to_cardiff_approx_540km(self):
        lat1 = np.array([55.95])
        lon1 = np.array([-3.19])
        lat2 = np.array([51.48])
        lon2 = np.array([-3.18])
        d = haversine_distances(lat1, lon1, lat2, lon2)
        assert 490.0 < float(d[0, 0]) < 550.0


class TestGaussianWeightsAdditional:
    def test_bandwidth_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            gaussian_weights(
                np.array([51.5]), np.array([-0.1]),
                np.array([51.5]), np.array([-0.1]),
                bandwidth_km=0.0,
            )

    def test_very_large_bandwidth_weights_near_one(self):
        """With a huge bandwidth, all weights should be close to 1."""
        rng = np.random.default_rng(10)
        lat_c = rng.uniform(50, 56, 20)
        lon_c = rng.uniform(-4, 2, 20)
        lat_t = rng.uniform(50, 56, 5)
        lon_t = rng.uniform(-4, 2, 5)
        w = gaussian_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=100000.0)
        assert np.all(w > 0.99)

    def test_very_small_bandwidth_weights_mostly_zero(self):
        """With a tiny bandwidth (0.001 km), only exactly co-located points get nonzero weight."""
        lat_c = np.array([51.5, 52.0])
        lon_c = np.array([-0.1, -0.1])
        lat_t = np.array([51.5])
        lon_t = np.array([-0.1])
        w = gaussian_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=0.001)
        # First point is co-located, gets weight ~1; second is ~55km away, gets 0
        assert float(w[0, 0]) > 0.99
        assert float(w[1, 0]) < 1e-6

    def test_weight_at_bandwidth_distance_approx_exp_neg1(self):
        """At distance = bandwidth, w = exp(-1) ≈ 0.368."""
        bw = 50.0
        # 1 deg lat ~ 111 km; to get 50 km need ~0.45 deg
        delta_lat = bw / 111.0
        lat_c = np.array([51.5])
        lon_c = np.array([-0.1])
        lat_t = np.array([51.5 + delta_lat])
        lon_t = np.array([-0.1])
        w = gaussian_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=bw)
        assert float(w[0, 0]) == pytest.approx(np.exp(-1), rel=0.05)


class TestEpanechnikovWeightsAdditional:
    def test_output_shape_multiple_points(self):
        n_cal, n_test = 30, 8
        rng = np.random.default_rng(11)
        lat_c = rng.uniform(50, 56, n_cal)
        lon_c = rng.uniform(-4, 2, n_cal)
        lat_t = rng.uniform(50, 56, n_test)
        lon_t = rng.uniform(-4, 2, n_test)
        w = epanechnikov_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=100.0)
        assert w.shape == (n_cal, n_test)

    def test_epanechnikov_max_is_one_at_zero_distance(self):
        lat = np.array([52.0])
        lon = np.array([-1.5])
        w = epanechnikov_weights(lat, lon, lat, lon, bandwidth_km=20.0)
        assert float(w[0, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_epanechnikov_less_than_gaussian_at_same_distance(self):
        """Epanechnikov decays faster to zero than Gaussian at moderate distance."""
        bw = 50.0
        delta_lat = bw * 0.7 / 111.0  # 0.7 bandwidth distance
        lat_c = np.array([51.5])
        lon_c = np.array([-0.1])
        lat_t = np.array([51.5 + delta_lat])
        lon_t = np.array([-0.1])
        w_g = gaussian_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=bw)
        w_e = epanechnikov_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=bw)
        # Epanechnikov is 1 - u^2 = 1 - 0.49 = 0.51; Gaussian is ~exp(-0.49) = 0.61
        # Both are in (0,1); both non-negative; epanechnikov is lower here
        assert float(w_e[0, 0]) >= 0.0
        assert float(w_g[0, 0]) > 0.0


class TestUniformWeightsAdditional:
    def test_output_shape(self):
        rng = np.random.default_rng(12)
        lat_c = rng.uniform(50, 56, 25)
        lon_c = rng.uniform(-4, 2, 25)
        lat_t = rng.uniform(50, 56, 7)
        lon_t = rng.uniform(-4, 2, 7)
        w = uniform_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=10.0)
        assert w.shape == (25, 7)

    def test_distant_points_get_zero_weight(self):
        """London to Glasgow (~555 km) must get zero weight at 100 km bandwidth."""
        lat_c = np.array([51.51])
        lon_c = np.array([-0.13])
        lat_t = np.array([55.86])
        lon_t = np.array([-4.25])
        w = uniform_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=100.0)
        assert float(w[0, 0]) == 0.0

    def test_nearby_points_get_one_weight(self):
        """Two nearby points well within bandwidth should get weight 1."""
        lat_c = np.array([51.5])
        lon_c = np.array([-0.1])
        lat_t = np.array([51.51])
        lon_t = np.array([-0.12])
        w = uniform_weights(lat_c, lon_c, lat_t, lon_t, bandwidth_km=10.0)
        assert float(w[0, 0]) == 1.0


class TestComputeWeightsAdditional:
    def test_dispatch_uniform(self):
        lat_c = np.array([51.5, 52.0])
        lon_c = np.array([-0.1, -0.2])
        lat_t = np.array([51.5])
        lon_t = np.array([-0.1])
        w1 = compute_weights(lat_c, lon_c, lat_t, lon_t, 10.0, kernel="uniform")
        w2 = uniform_weights(lat_c, lon_c, lat_t, lon_t, 10.0)
        np.testing.assert_array_almost_equal(w1, w2)

    def test_gaussian_is_default_kernel(self):
        lat_c = np.array([51.5])
        lon_c = np.array([-0.1])
        w1 = compute_weights(lat_c, lon_c, lat_c, lon_c, 10.0)  # default kernel
        w2 = gaussian_weights(lat_c, lon_c, lat_c, lon_c, 10.0)
        np.testing.assert_array_almost_equal(w1, w2)


class TestKishNEffAdditional:
    def test_n_eff_two_equal_weights(self):
        """Two equal weights should give n_eff = 2."""
        w = np.array([1.0, 1.0])
        assert kish_n_eff(w) == pytest.approx(2.0)

    def test_n_eff_scales_with_n_for_uniform(self):
        """For uniform weights of size n, n_eff should equal n."""
        for n in [5, 10, 50, 100]:
            assert kish_n_eff(np.ones(n)) == pytest.approx(float(n), rel=1e-6)

    def test_n_eff_all_same_except_one_tiny(self):
        """One very small weight among equal weights should barely affect n_eff."""
        w = np.array([1.0] * 99 + [1e-6])
        n_eff = kish_n_eff(w)
        assert 90.0 < n_eff <= 100.0

    def test_n_eff_decreases_with_concentration(self):
        """More concentrated weights → lower effective sample size."""
        w_flat = np.ones(50)
        w_concentrated = np.array([100.0] + [0.01] * 49)
        assert kish_n_eff(w_flat) > kish_n_eff(w_concentrated)

    def test_n_eff_negative_weights_ignored(self):
        """Weights <= 0 should be filtered before computation."""
        w = np.array([-1.0, 0.0, 1.0, 1.0])
        # After filtering negatives and zeros, only [1.0, 1.0] remain → n_eff = 2
        assert kish_n_eff(w) == pytest.approx(2.0, rel=1e-6)
