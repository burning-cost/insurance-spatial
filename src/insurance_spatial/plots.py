"""
Plotting utilities for spatial territory models.

Optional dependency: matplotlib (and geopandas for choropleth maps).

Install with: uv add insurance-spatial[plots]

All functions return matplotlib Figure objects so you can save or display them
in whatever context you are working in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import matplotlib.figure


def plot_relativities(
    relativities: pl.DataFrame,
    area_col: str = "area",
    rel_col: str = "relativity",
    lower_col: str = "lower",
    upper_col: str = "upper",
    title: str = "Territory Relativities",
    n_areas: int = 40,
    figsize: tuple[int, int] = (12, 6),
) -> "matplotlib.figure.Figure":
    """
    Bar plot of territory relativities with credibility intervals.

    Shows the top/bottom n_areas by relativity value so the plot stays readable.

    Parameters
    ----------
    relativities :
        Polars DataFrame from BYM2Result.territory_relativities().
    area_col, rel_col, lower_col, upper_col :
        Column names in the DataFrame.
    title :
        Plot title.
    n_areas :
        Number of areas to show (highest and lowest by relativity).
    figsize :
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "plot_relativities requires matplotlib. uv add insurance-spatial[plots]"
        ) from exc

    df = relativities.sort(rel_col, descending=True)
    n = min(n_areas, len(df))
    # Take top n/2 and bottom n/2
    half = n // 2
    top = df.head(half)
    bottom = df.tail(half)
    display_df = pl.concat([top, bottom])

    areas = display_df[area_col].to_list()
    rels = np.array(display_df[rel_col].to_list())
    lowers = np.array(display_df[lower_col].to_list())
    uppers = np.array(display_df[upper_col].to_list())

    err_low = rels - lowers
    err_high = uppers - rels

    fig, ax = plt.subplots(figsize=figsize)
    colours = ["#d62728" if r > 1.0 else "#1f77b4" for r in rels]
    x = np.arange(len(areas))
    ax.bar(x, rels - 1.0, bottom=1.0, color=colours, alpha=0.7, width=0.6)
    ax.errorbar(
        x, rels,
        yerr=[err_low, err_high],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=0.8,
    )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="Reference (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Relativity (multiplicative)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_trace(
    result: object,
    params: Optional[list[str]] = None,
    figsize: tuple[int, int] = (10, 8),
) -> "matplotlib.figure.Figure":
    """
    Trace plot for MCMC convergence inspection.

    Uses ArviZ's plot_trace internally.

    Parameters
    ----------
    result :
        BYM2Result object.
    params :
        List of parameter names to plot.  Defaults to ["alpha", "sigma", "rho"].
    figsize :
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import arviz as az
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_trace requires arviz and matplotlib. uv add arviz matplotlib"
        ) from exc

    if params is None:
        params = ["alpha", "sigma", "rho"]

    axes = az.plot_trace(result.trace, var_names=params, figsize=figsize)
    fig = axes.ravel()[0].get_figure()
    return fig


def plot_choropleth(
    relativities: pl.DataFrame,
    geodataframe,  # geopandas.GeoDataFrame
    merge_on_rel: str = "area",
    merge_on_geo: str = "area",
    rel_col: str = "relativity",
    title: str = "Territory Relativities",
    cmap: str = "RdYlGn_r",
    figsize: tuple[int, int] = (10, 12),
) -> "matplotlib.figure.Figure":
    """
    Choropleth map of territory relativities.

    Parameters
    ----------
    relativities :
        Polars DataFrame from BYM2Result.territory_relativities().
    geodataframe :
        geopandas.GeoDataFrame with polygon geometries for each area.
    merge_on_rel :
        Column in relativities to merge on.
    merge_on_geo :
        Column in geodataframe to merge on.
    rel_col :
        Column name in relativities containing the relativity values.
    title :
        Map title.
    cmap :
        Matplotlib colourmap.  ``RdYlGn_r`` (red = high risk, green = low risk)
        is conventional for insurance territory maps.
    figsize :
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "plot_choropleth requires geopandas and matplotlib. "
            "uv add insurance-spatial[plots]"
        ) from exc

    # Merge relativities into the GeoDataFrame
    rel_pandas = relativities.select([merge_on_rel, rel_col]).to_pandas()
    merged = geodataframe.merge(
        rel_pandas,
        left_on=merge_on_geo,
        right_on=merge_on_rel,
        how="left",
    )

    fig, ax = plt.subplots(figsize=figsize)
    merged.plot(
        column=rel_col,
        ax=ax,
        cmap=cmap,
        legend=True,
        legend_kwds={"label": "Relativity", "orientation": "horizontal"},
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig
