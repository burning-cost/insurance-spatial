"""
Postcode to coordinate conversion for UK pricing data.

UK pricing teams work with postcodes, not lat/lon. This module converts postcode
strings to WGS-84 coordinates using the pgeocode library (which wraps GeoNames
data). It handles the two common failure modes in UK pricing data:
  1. Full postcodes that aren't in the database (new builds, recent changes)
  2. Outward-code-only postcodes (e.g. 'SW1A' without the inward code)

The fallback strategy: outward code centroids are precomputed from the full
database, so 'SW1A 2AA' failing gets mapped to the centroid of 'SW1A'.

Dependencies: pgeocode >= 0.4
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np


def _normalise_postcode(pc: str) -> str:
    """Upper-case and strip internal whitespace."""
    return re.sub(r"\s+", " ", pc.strip().upper())


def _outward_code(pc: str) -> str:
    """
    Extract outward code (first part before the space).

    Examples
    --------
    >>> _outward_code('SW1A 2AA')
    'SW1A'
    >>> _outward_code('EX39')
    'EX39'
    """
    parts = pc.strip().split()
    return parts[0] if parts else pc.strip()


class PostcodeGeocoder:
    """
    Postcode-to-coordinate lookup for Great Britain.

    Wraps pgeocode with caching, bulk vectorised lookup, and graceful fallback
    to outward-code centroids when a full postcode is not found.

    Parameters
    ----------
    country : str, default 'GB'
        ISO country code. GB covers England, Scotland, and Wales.
        Use 'IE' for Republic of Ireland postcodes.

    Examples
    --------
    >>> gc = PostcodeGeocoder()
    >>> lat, lon = gc.geocode(['SW1A 2AA', 'EX39 1LB', 'G1 1XQ'])
    >>> lat
    array([51.5009, 51.0476, 55.8617])

    >>> # Handles outward-code-only inputs
    >>> lat, lon = gc.geocode(['SW1A', 'EX39'])
    """

    def __init__(self, country: str = "GB") -> None:
        try:
            import pgeocode
        except ImportError as exc:
            raise ImportError(
                "pgeocode is required for PostcodeGeocoder. "
                "Install with: pip install pgeocode"
            ) from exc

        self._pgeocode = pgeocode
        self.country = country.upper()
        self._nomi: Optional[object] = None  # lazy init
        self._outward_cache: dict[str, tuple[float, float]] = {}

    def _get_nomi(self) -> object:
        """Lazy-initialise the pgeocode Nominatim object (downloads data on first call)."""
        if self._nomi is None:
            self._nomi = self._pgeocode.Nominatim(self.country)
        return self._nomi

    def _outward_centroid(self, outward: str) -> tuple[float, float]:
        """
        Centroid of all postcodes with the given outward code.

        Computed once per outward code and cached.
        """
        if outward in self._outward_cache:
            return self._outward_cache[outward]

        nomi = self._get_nomi()
        # pgeocode's internal dataframe — may vary across versions
        df = nomi._data  # type: ignore[attr-defined]
        if df is None or len(df) == 0:
            return (float("nan"), float("nan"))

        # Try to find rows matching the outward code
        col = "postal_code"
        if col not in df.columns:
            return (float("nan"), float("nan"))

        mask = df[col].str.startswith(outward, na=False)
        subset = df[mask]
        if len(subset) == 0:
            result = (float("nan"), float("nan"))
        else:
            lat = float(subset["latitude"].dropna().mean())
            lon = float(subset["longitude"].dropna().mean())
            result = (lat, lon)

        self._outward_cache[outward] = result
        return result

    def geocode(
        self, postcodes: list[str] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a list of postcodes to (lat, lon) arrays.

        Missing or invalid postcodes fall back to the outward-code centroid.
        If even that fails, NaN is returned for that entry — the caller should
        handle NaN coordinates (typically by dropping those rows or using a
        regional default).

        Parameters
        ----------
        postcodes : list of str or array-like
            UK postcode strings. Full postcodes (e.g. 'SW1A 2AA') preferred;
            outward codes (e.g. 'SW1A') are accepted.

        Returns
        -------
        lat : np.ndarray, shape (n,)
            Latitude in decimal degrees (WGS-84).
        lon : np.ndarray, shape (n,)
            Longitude in decimal degrees (WGS-84).

        Examples
        --------
        >>> gc = PostcodeGeocoder()
        >>> lat, lon = gc.geocode(['SW1A 2AA', 'M1 1AE'])
        """
        postcodes = [_normalise_postcode(str(pc)) for pc in postcodes]
        nomi = self._get_nomi()

        result = nomi.query_postal_code(postcodes)

        lat_arr = np.asarray(result["latitude"], dtype=float)
        lon_arr = np.asarray(result["longitude"], dtype=float)

        n_missing = int(np.sum(np.isnan(lat_arr)))
        if n_missing > 0:
            # Fall back to outward-code centroid for missing entries
            for i, (la, pc) in enumerate(zip(lat_arr, postcodes)):
                if np.isnan(la):
                    outward = _outward_code(pc)
                    clat, clon = self._outward_centroid(outward)
                    lat_arr[i] = clat
                    lon_arr[i] = clon

            n_still_missing = int(np.sum(np.isnan(lat_arr)))
            if n_still_missing > 0:
                warnings.warn(
                    f"{n_still_missing} postcode(s) could not be geocoded (including "
                    "outward-code fallback). These will have NaN coordinates. "
                    "Consider providing explicit lat/lon for these rows.",
                    UserWarning,
                    stacklevel=2,
                )

        return lat_arr, lon_arr

    def geocode_with_flags(
        self, postcodes: list[str] | np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Geocode with a flag array indicating which lookups used fallback logic.

        Returns
        -------
        dict with keys: 'lat', 'lon', 'used_outward_fallback', 'is_missing'
        """
        postcodes_norm = [_normalise_postcode(str(pc)) for pc in postcodes]
        nomi = self._get_nomi()
        result = nomi.query_postal_code(postcodes_norm)

        lat_arr = np.asarray(result["latitude"], dtype=float)
        lon_arr = np.asarray(result["longitude"], dtype=float)
        used_fallback = np.zeros(len(lat_arr), dtype=bool)

        for i, (la, pc) in enumerate(zip(lat_arr, postcodes_norm)):
            if np.isnan(la):
                outward = _outward_code(pc)
                clat, clon = self._outward_centroid(outward)
                lat_arr[i] = clat
                lon_arr[i] = clon
                used_fallback[i] = True

        is_missing = np.isnan(lat_arr)
        return {
            "lat": lat_arr,
            "lon": lon_arr,
            "used_outward_fallback": used_fallback,
            "is_missing": is_missing,
        }
