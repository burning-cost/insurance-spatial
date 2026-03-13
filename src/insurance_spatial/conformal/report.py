"""
SpatialCoverageReport: diagnostic coverage analysis for spatially weighted conformal predictors.

The fundamental diagnostic question is: does my predictor cover 90% of risks in
every part of the country, or just 90% nationally? A model can be nationally
correct and geographically broken. This report tells you which.

The MACG (Mean Absolute Coverage Gap) metric aggregates per-cell coverage gaps
across a spatial grid. Lower is better. A well-calibrated spatial predictor
should have MACG < 0.02 at 20x20 resolution; standard conformal typically
produces MACG of 0.05-0.15 on geographically concentrated portfolios.

The FCA consumer duty table is specifically designed for UK regulatory reporting:
you need to demonstrate that pricing model uncertainty is not disproportionately
concentrated in regions with protected characteristics (e.g. deprived areas, flood
zones). Coverage gaps in these areas create conduct risk.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import polars as pl

from insurance_spatial.conformal._types import CoverageResult


class SpatialCoverageReport:
    """
    Diagnostic coverage analysis for a calibrated SpatialConformalPredictor.

    Parameters
    ----------
    predictor : SpatialConformalPredictor
        A calibrated predictor (calibrate() must have been called).

    Examples
    --------
    >>> report = SpatialCoverageReport(predictor)
    >>> result = report.evaluate(X_val, y_val, lat=lat_val, lon=lon_val)
    >>> print(f"MACG: {result.macg:.4f}, marginal coverage: {result.marginal_coverage:.3f}")
    >>> fig = report.coverage_map(resolution=20)
    >>> fig.savefig("coverage_map.png", dpi=150)
    """

    def __init__(self, predictor: Any) -> None:
        if not predictor.is_calibrated_:
            raise RuntimeError(
                "Predictor must be calibrated before creating a SpatialCoverageReport. "
                "Call predictor.calibrate() first."
            )
        self.predictor = predictor
        self._result: Optional[CoverageResult] = None
        self._intervals: Optional[Any] = None
        self._y_val: Optional[np.ndarray] = None
        self._lat_val: Optional[np.ndarray] = None
        self._lon_val: Optional[np.ndarray] = None

    def evaluate(
        self,
        X_val: Any,
        y_val: np.ndarray,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        postcodes: Optional[list[str]] = None,
        alpha: float = 0.10,
        grid_resolution: int = 20,
    ) -> CoverageResult:
        """
        Evaluate spatial coverage on a labelled validation set.

        Parameters
        ----------
        X_val : array-like
            Feature matrix for the validation set.
        y_val : array-like, shape (n_val,)
            Observed outcomes on the validation set.
        lat : np.ndarray, optional
        lon : np.ndarray, optional
        postcodes : list of str, optional
        alpha : float, default 0.10
        grid_resolution : int, default 20
            Grid cells per side for MACG calculation. 20 gives a 20x20 = 400 cell grid,
            matching the Hjort et al. benchmark.

        Returns
        -------
        CoverageResult
        """
        y_val = np.asarray(y_val, dtype=float)

        # Resolve coordinates
        if lat is not None and lon is not None:
            lat_arr = np.asarray(lat, dtype=float)
            lon_arr = np.asarray(lon, dtype=float)
        elif postcodes is not None:
            from insurance_spatial.conformal.geocoder import PostcodeGeocoder
            gc = PostcodeGeocoder()
            lat_arr, lon_arr = gc.geocode(postcodes)
        else:
            raise ValueError("Must provide lat/lon or postcodes for validation set.")

        intervals = self.predictor.predict_interval(
            X_val, lat=lat_arr, lon=lon_arr, alpha=alpha
        )

        covered = (y_val >= intervals.lower) & (y_val <= intervals.upper)
        marginal_cov = float(np.mean(covered))

        # MACG on spatial grid
        macg, cell_lats, cell_lons, cell_covs = self._compute_macg(
            covered, lat_arr, lon_arr, alpha, grid_resolution
        )

        result = CoverageResult(
            alpha=alpha,
            target_coverage=1.0 - alpha,
            marginal_coverage=marginal_cov,
            macg=macg,
            n_grid_cells=len(cell_lats) if cell_lats is not None else 0,
            n_val=len(y_val),
            coverage_by_cell=cell_covs,
            cell_centres_lat=cell_lats,
            cell_centres_lon=cell_lons,
        )

        self._result = result
        self._intervals = intervals
        self._y_val = y_val
        self._lat_val = lat_arr
        self._lon_val = lon_arr

        return result

    def _compute_macg(
        self,
        covered: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        alpha: float,
        resolution: int,
    ) -> tuple[float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute MACG over a regular grid."""
        lat_min, lat_max = lat.min(), lat.max()
        lon_min, lon_max = lon.min(), lon.max()

        dlat = max((lat_max - lat_min) * 0.02, 0.005)
        dlon = max((lon_max - lon_min) * 0.02, 0.005)

        lat_edges = np.linspace(lat_min - dlat, lat_max + dlat, resolution + 1)
        lon_edges = np.linspace(lon_min - dlon, lon_max + dlon, resolution + 1)

        target = 1.0 - alpha
        gaps = []
        cell_lat_centres = []
        cell_lon_centres = []
        cell_coverages = []

        for i in range(resolution):
            for j in range(resolution):
                mask = (
                    (lat >= lat_edges[i]) & (lat < lat_edges[i + 1]) &
                    (lon >= lon_edges[j]) & (lon < lon_edges[j + 1])
                )
                n_cell = int(np.sum(mask))
                if n_cell < 3:
                    continue

                cell_cov = float(np.mean(covered[mask]))
                gaps.append(abs(target - cell_cov))
                cell_lat_centres.append(0.5 * (lat_edges[i] + lat_edges[i + 1]))
                cell_lon_centres.append(0.5 * (lon_edges[j] + lon_edges[j + 1]))
                cell_coverages.append(cell_cov)

        if not gaps:
            return float("nan"), None, None, None

        macg = float(np.mean(gaps))
        return (
            macg,
            np.array(cell_lat_centres),
            np.array(cell_lon_centres),
            np.array(cell_coverages),
        )

    def coverage_map(
        self,
        resolution: int = 20,
        figsize: tuple[float, float] = (10, 8),
        title: Optional[str] = None,
        cmap: str = "RdYlGn",
    ) -> "matplotlib.figure.Figure":
        """
        Matplotlib figure showing empirical coverage by spatial grid cell.

        Green cells are near the target coverage; red cells are under- or
        over-covered. The colour scale is centred on the target coverage.

        Parameters
        ----------
        resolution : int, default 20
        figsize : tuple, default (10, 8)
        title : str, optional
        cmap : str, default 'RdYlGn'

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If evaluate() has not been called.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        self._check_evaluated()
        result = self._result

        if result.coverage_by_cell is None:
            raise ValueError("No cell coverage data available.")

        target = result.target_coverage
        vmin = max(0.0, target - 0.20)
        vmax = min(1.0, target + 0.20)

        fig, ax = plt.subplots(figsize=figsize)

        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=target, vmax=vmax)
        sc = ax.scatter(
            result.cell_centres_lon,
            result.cell_centres_lat,
            c=result.coverage_by_cell,
            cmap=cmap,
            norm=norm,
            s=max(20, 3000 / resolution ** 1.5),
            marker="s",
            alpha=0.85,
        )

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f"Empirical coverage (target = {target:.0%})", fontsize=11)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            title or (
                f"Spatial coverage map — alpha={result.alpha:.0%}, "
                f"MACG={result.macg:.4f}"
            )
        )

        # Annotate marginal coverage
        ax.text(
            0.02, 0.02,
            f"Marginal coverage: {result.marginal_coverage:.3f}\n"
            f"MACG ({result.n_grid_cells} cells): {result.macg:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        fig.tight_layout()
        return fig

    def fca_consumer_duty_table(
        self,
        region_labels: Optional[np.ndarray] = None,
        region_name: str = "Region",
    ) -> pl.DataFrame:
        """
        Coverage breakdown by geographic region for FCA Consumer Duty reporting.

        Under Consumer Duty (PS22/9), firms must demonstrate good outcomes across
        customer segments. Systematic under-coverage in deprived areas or rural
        postcodes is a conduct risk that must be identified and addressed.

        Parameters
        ----------
        region_labels : np.ndarray of str, optional, shape (n_val,)
            Region label per validation observation (e.g. county, flood zone,
            IMD quintile, London/outside). If None, uses outward postcode
            as a rough proxy.
        region_name : str, default 'Region'
            Column name for the region in the output table.

        Returns
        -------
        pl.DataFrame
            Columns: region, n_policies, coverage, target_coverage, coverage_gap, flag

        Raises
        ------
        RuntimeError
            If evaluate() has not been called.

        Examples
        --------
        >>> table = report.fca_consumer_duty_table(region_labels=county_labels)
        >>> print(table.filter(pl.col("flag") == "REVIEW"))
        """
        self._check_evaluated()

        result = self._result
        target = result.target_coverage

        covered = (self._y_val >= self._intervals.lower) & (
            self._y_val <= self._intervals.upper
        )

        if region_labels is None:
            # Fall back: label by latitude band as a rough geographic proxy
            lat_bands = np.linspace(
                self._lat_val.min(), self._lat_val.max(), 6
            )
            band_idx = np.digitize(self._lat_val, lat_bands) - 1
            band_idx = np.clip(band_idx, 0, 4)
            region_labels = np.array([
                ["South", "South Midlands", "Midlands", "North Midlands", "North"][i]
                for i in band_idx
            ])

        # Aggregate
        rows = []
        unique_regions = np.unique(region_labels)
        for region in unique_regions:
            mask = region_labels == region
            n = int(np.sum(mask))
            if n == 0:
                continue
            cov = float(np.mean(covered[mask]))
            gap = target - cov
            if abs(gap) > 0.05:
                flag = "REVIEW"
            elif abs(gap) > 0.025:
                flag = "MONITOR"
            else:
                flag = "OK"
            rows.append({
                region_name: str(region),
                "n_policies": n,
                "coverage": round(cov, 4),
                "target_coverage": round(target, 4),
                "coverage_gap": round(gap, 4),
                "flag": flag,
            })

        df = pl.DataFrame(rows).sort("coverage_gap", descending=True)
        return df

    def macg_by_region(
        self, region_labels: np.ndarray, region_name: str = "Region"
    ) -> pl.DataFrame:
        """
        MACG broken down by pre-defined region labels.

        More informative than cell-by-cell MACG when you have meaningful geographic
        segments (counties, pricing territories, flood zones).

        Parameters
        ----------
        region_labels : np.ndarray of str, shape (n_val,)
        region_name : str

        Returns
        -------
        pl.DataFrame
            Columns: region, n_policies, macg, marginal_coverage, n_cells
        """
        self._check_evaluated()

        result = self._result
        target = result.target_coverage
        covered = (self._y_val >= self._intervals.lower) & (
            self._y_val <= self._intervals.upper
        )

        rows = []
        for region in np.unique(region_labels):
            mask = region_labels == region
            n = int(np.sum(mask))
            if n < 5:
                continue

            lat_r = self._lat_val[mask]
            lon_r = self._lon_val[mask]
            cov_r = covered[mask]

            # Mini-grid MACG for this region
            macg, _, _, _ = self._compute_macg(cov_r, lat_r, lon_r, result.alpha, 5)
            marginal = float(np.mean(cov_r))

            rows.append({
                region_name: str(region),
                "n_policies": n,
                "marginal_coverage": round(marginal, 4),
                "macg": round(float(macg) if not np.isnan(float(macg)) else 0.0, 4),
            })

        return pl.DataFrame(rows).sort("macg", descending=True)

    def summary(self) -> str:
        """
        Plain-text summary of the coverage evaluation.

        Returns
        -------
        str
        """
        self._check_evaluated()
        r = self._result
        lines = [
            "=== Spatial Coverage Report ===",
            f"  Validation set: {r.n_val:,} observations",
            f"  Target coverage (1-alpha): {r.target_coverage:.1%}",
            f"  Marginal coverage: {r.marginal_coverage:.3f}",
            f"  Coverage gap: {r.coverage_gap():+.4f}",
            f"  MACG ({r.n_grid_cells} grid cells): {r.macg:.4f}",
            f"  Bandwidth: {self.predictor.bandwidth_km_:.1f} km",
            f"  Kernel: {self.predictor.spatial_kernel}",
        ]
        return "\n".join(lines)

    def _check_evaluated(self) -> None:
        if self._result is None:
            raise RuntimeError(
                "Report has not been evaluated. Call evaluate() first."
            )
