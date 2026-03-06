"""
Tests for the adjacency module.

These tests run locally: no model fitting, no heavy computation.
"""

import numpy as np
import pytest
from scipy import sparse

from insurance_spatial.adjacency import (
    AdjacencyMatrix,
    build_grid_adjacency,
    compute_bym2_scaling_factor,
)


class TestBuildGridAdjacency:
    def test_shape_rook(self):
        adj = build_grid_adjacency(3, 4, connectivity="rook")
        assert adj.W.shape == (12, 12)
        assert len(adj.areas) == 12

    def test_shape_queen(self):
        adj = build_grid_adjacency(4, 4, connectivity="queen")
        assert adj.W.shape == (16, 16)
        assert len(adj.areas) == 16

    def test_area_labels(self):
        adj = build_grid_adjacency(2, 3, connectivity="rook")
        assert adj.areas == ["r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"]

    def test_symmetry(self):
        adj = build_grid_adjacency(4, 4, connectivity="rook")
        W = adj.to_dense()
        np.testing.assert_array_equal(W, W.T)

    def test_no_self_loops(self):
        adj = build_grid_adjacency(4, 4, connectivity="rook")
        W = adj.to_dense()
        np.testing.assert_array_equal(np.diag(W), np.zeros(16))

    def test_rook_corner_has_2_neighbours(self):
        # Top-left corner of a 4×4 grid should have exactly 2 rook neighbours
        adj = build_grid_adjacency(4, 4, connectivity="rook")
        W = adj.to_dense()
        corner_idx = 0  # r0c0
        assert int(W[corner_idx].sum()) == 2

    def test_rook_interior_has_4_neighbours(self):
        # Centre node of a 5×5 grid (r2c2, index 12) has 4 rook neighbours
        adj = build_grid_adjacency(5, 5, connectivity="rook")
        W = adj.to_dense()
        centre_idx = 2 * 5 + 2  # = 12
        assert int(W[centre_idx].sum()) == 4

    def test_queen_interior_has_8_neighbours(self):
        adj = build_grid_adjacency(5, 5, connectivity="queen")
        W = adj.to_dense()
        centre_idx = 2 * 5 + 2
        assert int(W[centre_idx].sum()) == 8

    def test_is_csr_sparse(self):
        adj = build_grid_adjacency(3, 3)
        assert isinstance(adj.W, sparse.csr_matrix)

    def test_binary_weights(self):
        adj = build_grid_adjacency(4, 4)
        W = adj.to_dense()
        assert set(np.unique(W)).issubset({0.0, 1.0})

    def test_connected_single_component(self):
        adj = build_grid_adjacency(5, 5)
        assert adj.n_components() == 1


class TestAdjacencyMatrix:
    def test_n_property(self):
        adj = build_grid_adjacency(3, 3)
        assert adj.n == 9

    def test_to_edge_list(self):
        adj = build_grid_adjacency(2, 2, connectivity="rook")
        edges = adj.to_edge_list()
        assert edges.shape[1] == 2
        # 2×2 rook grid has 4 edges (2 horizontal + 2 vertical)
        assert edges.shape[0] == 4
        # All i < j
        assert np.all(edges[:, 0] < edges[:, 1])

    def test_neighbour_counts_rook_3x3(self):
        adj = build_grid_adjacency(3, 3, connectivity="rook")
        counts = adj.neighbour_counts()
        # corners: 2, edges: 3, centre: 4
        assert counts[0] == 2   # r0c0 corner
        assert counts[4] == 4   # r1c1 centre
        assert counts[1] == 3   # r0c1 edge

    def test_area_index_roundtrip(self):
        adj = build_grid_adjacency(3, 3)
        idx = adj.area_index()
        for area, i in idx.items():
            assert adj.areas[i] == area

    def test_wrong_shape_raises(self):
        W = sparse.eye(5, format="csr")
        with pytest.raises(ValueError, match="shape"):
            AdjacencyMatrix(W=W, areas=["a", "b", "c"])  # 3 labels, 5×5 matrix

    def test_dense_roundtrip(self):
        adj = build_grid_adjacency(3, 3)
        W_dense = adj.to_dense()
        assert W_dense.shape == (9, 9)
        assert W_dense.dtype == np.float64


class TestScalingFactor:
    def test_returns_positive_float(self):
        adj = build_grid_adjacency(3, 3)
        s = compute_bym2_scaling_factor(adj.W)
        assert isinstance(s, float)
        assert s > 0

    def test_finite(self):
        adj = build_grid_adjacency(5, 5)
        s = adj.scaling_factor
        assert np.isfinite(s)
        assert s > 0

    def test_cached_on_property(self):
        adj = build_grid_adjacency(3, 3)
        s1 = adj.scaling_factor
        s2 = adj.scaling_factor
        assert s1 == s2

    def test_scaling_factor_increases_with_grid_size(self):
        """
        For regular grids, the ICAR marginal variances scale with graph size.
        A larger grid should have a larger scaling factor.
        """
        adj_small = build_grid_adjacency(3, 3)
        adj_large = build_grid_adjacency(5, 5)
        assert adj_large.scaling_factor > adj_small.scaling_factor

    def test_larger_grid_different_from_smaller(self):
        adj_small = build_grid_adjacency(3, 3)
        adj_large = build_grid_adjacency(5, 5)
        # Scaling factors should differ (they depend on graph topology)
        assert abs(adj_small.scaling_factor - adj_large.scaling_factor) > 1e-6
