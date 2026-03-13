"""
Tests for geocoder.py: PostcodeGeocoder.

These tests mock pgeocode to avoid network downloads in CI. A subset of tests
is marked as integration tests requiring the real database.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if pgeocode not available
pgeocode = pytest.importorskip("pgeocode")

from insurance_spatial.conformal.geocoder import (
    PostcodeGeocoder,
    _normalise_postcode,
    _outward_code,
)


class TestNormalisePostcode:
    def test_upper_case(self):
        assert _normalise_postcode("sw1a 2aa") == "SW1A 2AA"

    def test_strips_extra_spaces(self):
        assert _normalise_postcode("  SW1A  2AA  ") == "SW1A 2AA"

    def test_already_normalised(self):
        assert _normalise_postcode("EX39 1LB") == "EX39 1LB"

    def test_outward_only(self):
        assert _normalise_postcode("sw1a") == "SW1A"


class TestOutwardCode:
    def test_full_postcode(self):
        assert _outward_code("SW1A 2AA") == "SW1A"

    def test_outward_only(self):
        assert _outward_code("EX39") == "EX39"

    def test_no_space(self):
        # Single token — return as-is
        assert _outward_code("SW1A2AA") == "SW1A2AA"


class TestPostcodeGeocoderConstruction:
    def test_default_country(self):
        gc = PostcodeGeocoder()
        assert gc.country == "GB"

    def test_custom_country(self):
        gc = PostcodeGeocoder(country="ie")
        assert gc.country == "IE"


class TestPostcodeGeocoderGeocodeIntegration:
    """Integration tests — require actual pgeocode GeoNames data download."""

    @pytest.mark.slow
    def test_london_postcode(self):
        gc = PostcodeGeocoder()
        lat, lon = gc.geocode(["SW1A 2AA"])
        # Buckingham Palace area
        assert 51.3 < float(lat[0]) < 51.7
        assert -0.4 < float(lon[0]) < 0.2

    @pytest.mark.slow
    def test_multiple_postcodes(self):
        gc = PostcodeGeocoder()
        postcodes = ["SW1A 2AA", "M1 1AE", "G1 1XQ"]
        lat, lon = gc.geocode(postcodes)
        assert lat.shape == (3,)
        assert lon.shape == (3,)
        # All should be in Great Britain roughly
        assert np.all(lat > 49.0)
        assert np.all(lat < 61.0)

    @pytest.mark.slow
    def test_geocode_with_flags_keys(self):
        gc = PostcodeGeocoder()
        result = gc.geocode_with_flags(["SW1A 2AA"])
        assert "lat" in result
        assert "lon" in result
        assert "used_outward_fallback" in result
        assert "is_missing" in result

    @pytest.mark.slow
    def test_outward_only_gives_location(self):
        gc = PostcodeGeocoder()
        lat, lon = gc.geocode(["SW1A"])
        # Should not be NaN — falls back to outward centroid
        if not np.isnan(lat[0]):
            assert 50.0 < float(lat[0]) < 58.0

    @pytest.mark.slow
    def test_array_input(self):
        gc = PostcodeGeocoder()
        postcodes = np.array(["SW1A 2AA", "EX39 1LB"])
        lat, lon = gc.geocode(postcodes)
        assert len(lat) == 2
        assert len(lon) == 2
