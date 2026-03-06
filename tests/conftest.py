"""
Shared fixtures for insurance-spatial tests.

Synthetic data strategy
-----------------------
We use a 5×5 grid for all model tests.  It has 25 nodes, enough to exercise
the spatial model and its diagnostics while being small enough to run quickly
on Databricks serverless compute.

Claims are generated from a Poisson process with a known spatial pattern so
we can test that the model captures it qualitatively.
"""

import numpy as np
import pytest

from insurance_spatial.adjacency import build_grid_adjacency, AdjacencyMatrix


@pytest.fixture(scope="session")
def grid_5x5() -> AdjacencyMatrix:
    """5×5 rook-contiguity grid adjacency matrix."""
    return build_grid_adjacency(5, 5, connectivity="rook")


@pytest.fixture(scope="session")
def grid_3x3() -> AdjacencyMatrix:
    """3×3 rook-contiguity grid, for fast unit tests."""
    return build_grid_adjacency(3, 3, connectivity="rook")


@pytest.fixture(scope="session")
def synthetic_claims_5x5(grid_5x5: AdjacencyMatrix):
    """
    Synthetic Poisson claims data on the 5×5 grid.

    Returns (claims, exposure, true_log_rates).
    True log-rates have a north–south gradient (top row riskier) so that
    spatial autocorrelation is present.
    """
    rng = np.random.default_rng(2024)
    N = grid_5x5.n  # 25
    exposure = rng.uniform(50.0, 500.0, size=N)

    # True spatial pattern: top rows have higher risk
    row_idx = np.array([i // 5 for i in range(N)])  # 0..4
    true_log_rate = 0.5 - 0.2 * row_idx + 0.1 * rng.standard_normal(N)

    mu = exposure * np.exp(true_log_rate)
    claims = rng.poisson(mu)

    return claims, exposure, true_log_rate
