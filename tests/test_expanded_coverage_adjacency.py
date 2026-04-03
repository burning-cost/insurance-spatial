"""
Expanded test coverage for the adjacency module.

Targets: edge cases, boundary values, API contracts, and
untested methods on AdjacencyMatrix.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from insurance_spatial.adjacency import (
    AdjacencyMatrix,
    build_grid_adjacency,
    compute_bym2_scaling_factor,
    _connect_islands,
)


# ---------------------------------------------------------------------------
# AdjacencyMatrix construction and validation
# ---------------------------------------------------------------------------

class TestAdjacencyMatrixConstruction:
    def test_accepts_non_csr_sparse_converts_to_csr(self):
        """Non-CSR sparse matrices should be converted to CSR on init."""
        W_coo = sparse.coo_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        adj = AdjacencyMatrix(W=W_coo, areas=["a", "b"])
        assert isinstance(adj.W, sparse.csr_matrix)

    def test_accepts_lil_matrix(self):
        W_lil = sparse.lil_matrix((3, 3))
        W_lil[0, 1] = 1
        W_lil[1, 0] = 1
        W_lil[1, 2] = 1
        W_lil[2, 1] = 1
        adj = AdjacencyMatrix(W=W_lil.tocsr(), areas=["a", "b", "c"])
        assert adj.n == 3

    def test_wrong_shape_too_many_areas_raises(self):
        W = sparse.eye(4, format="csr")
        with pytest.raises(ValueError, match="shape"):
            AdjacencyMatrix(W=W, areas=["a", "b"])

    def test_wrong_shape_too_few_areas_raises(self):
        W = sparse.eye(2, format="csr")
        with pytest.raises(ValueError, match="shape"):
            AdjacencyMatrix(W=W, areas=["a", "b", "c", "d"])

    def test_single_node_adjacency(self):
        """A single isolated node is technically valid for the dataclass."""
        W = sparse.csr_matrix(np.array([[0.0]]))
        adj = AdjacencyMatrix(W=W, areas=["solo"])
        assert adj.n == 1

    def test_scaling_factor_precomputed_is_cached(self):
        """If _scaling_factor is passed directly, it should be returned as-is."""
        adj = build_grid_adjacency(3, 3)
        adj2 = AdjacencyMatrix(W=adj.W, areas=adj.areas, _scaling_factor=42.0)
        assert adj2.scaling_factor == 42.0

    def test_empty_areas_with_0x0_matrix(self):
        """Degenerate zero-area case."""
        W = sparse.csr_matrix((0, 0))
        adj = AdjacencyMatrix(W=W, areas=[])
        assert adj.n == 0


class TestAdjacencyMatrixMethods:
    def test_n_components_connected(self):
        adj = build_grid_adjacency(4, 4)
        assert adj.n_components() == 1

    def test_n_components_disconnected(self):
        """Two isolated nodes should give 2 components."""
        W = sparse.csr_matrix(np.zeros((2, 2)))
        adj = AdjacencyMatrix(W=W, areas=["a", "b"])
        assert adj.n_components() == 2

    def test_to_edge_list_1x1_grid_no_edges(self):
        """A 1x1 grid has no edges."""
        adj = build_grid_adjacency(1, 1)
        edges = adj.to_edge_list()
        assert edges.shape == (0, 2) or len(edges) == 0

    def test_to_edge_list_upper_triangle(self):
        """All returned edges must have i < j."""
        adj = build_grid_adjacency(5, 5, connectivity="queen")
        edges = adj.to_edge_list()
        if len(edges) > 0:
            assert np.all(edges[:, 0] < edges[:, 1])

    def test_to_edge_list_count_rook_2x3(self):
        """2x3 rook grid: edges = 2*2 horizontal + 1*3 vertical = 7."""
        adj = build_grid_adjacency(2, 3, connectivity="rook")
        edges = adj.to_edge_list()
        assert edges.shape[0] == 7

    def test_neighbour_counts_all_same_line_grid(self):
        """A 1xN line graph: endpoints have 1 neighbour, others have 2."""
        adj = build_grid_adjacency(1, 5, connectivity="rook")
        counts = adj.neighbour_counts()
        assert counts[0] == 1
        assert counts[4] == 1
        assert counts[2] == 2

    def test_area_index_returns_correct_mapping(self):
        adj = build_grid_adjacency(2, 2)
        idx = adj.area_index()
        assert len(idx) == 4
        for i, area in enumerate(adj.areas):
            assert idx[area] == i

    def test_to_dense_is_float64(self):
        adj = build_grid_adjacency(3, 3)
        W_dense = adj.to_dense()
        assert W_dense.dtype == np.float64

    def test_to_dense_symmetric(self):
        adj = build_grid_adjacency(4, 5, connectivity="queen")
        W = adj.to_dense()
        np.testing.assert_array_equal(W, W.T)


# ---------------------------------------------------------------------------
# build_grid_adjacency edge cases
# ---------------------------------------------------------------------------

class TestBuildGridAdjacencyEdgeCases:
    def test_1x1_grid_rook_no_neighbours(self):
        adj = build_grid_adjacency(1, 1, connectivity="rook")
        assert adj.W.nnz == 0
        assert adj.areas == ["r0c0"]

    def test_1x2_rook_single_edge(self):
        adj = build_grid_adjacency(1, 2, connectivity="rook")
        assert adj.W.nnz == 2  # symmetric: both directions

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError, match="connectivity"):
            build_grid_adjacency(3, 3, connectivity="bishop")

    def test_queen_has_more_edges_than_rook(self):
        adj_rook = build_grid_adjacency(4, 4, connectivity="rook")
        adj_queen = build_grid_adjacency(4, 4, connectivity="queen")
        assert adj_queen.W.nnz > adj_rook.W.nnz

    def test_large_grid_produces_correct_shape(self):
        adj = build_grid_adjacency(10, 10)
        assert adj.W.shape == (100, 100)
        assert len(adj.areas) == 100

    def test_area_labels_format(self):
        adj = build_grid_adjacency(3, 3)
        expected = [f"r{r}c{c}" for r in range(3) for c in range(3)]
        assert adj.areas == expected

    def test_single_row_grid_rook(self):
        """1×5 grid should be a path graph."""
        adj = build_grid_adjacency(1, 5, connectivity="rook")
        counts = adj.neighbour_counts()
        assert counts[0] == 1
        assert counts[1] == 2
        assert counts[4] == 1

    def test_single_col_grid_rook(self):
        """5×1 grid should be a path graph."""
        adj = build_grid_adjacency(5, 1, connectivity="rook")
        counts = adj.neighbour_counts()
        assert counts[0] == 1
        assert counts[2] == 2


# ---------------------------------------------------------------------------
# compute_bym2_scaling_factor
# ---------------------------------------------------------------------------

class TestComputeBym2ScalingFactorEdgeCases:
    def test_line_graph_3_nodes(self):
        """Path graph: 0--1--2, valid connected graph."""
        W = sparse.csr_matrix(np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float))
        s = compute_bym2_scaling_factor(W)
        assert s > 0
        assert np.isfinite(s)

    def test_complete_graph_3_nodes(self):
        """K3: all three nodes connected to each other."""
        W = sparse.csr_matrix(np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float))
        s = compute_bym2_scaling_factor(W)
        assert s > 0
        assert np.isfinite(s)

    def test_scaling_factor_is_symmetric_to_permutation(self):
        """Permuting rows/cols of W (relabelling areas) should not change s."""
        adj1 = build_grid_adjacency(3, 4)
        s1 = compute_bym2_scaling_factor(adj1.W)

        # Permute rows and cols together
        perm = [3, 0, 7, 1, 5, 11, 2, 4, 6, 8, 9, 10]
        W_arr = adj1.to_dense()
        W_perm = W_arr[np.ix_(perm, perm)]
        s2 = compute_bym2_scaling_factor(sparse.csr_matrix(W_perm))

        assert abs(s1 - s2) < 1e-6

    def test_known_value_2node_chain(self):
        """2-node path graph has analytical pseudoinverse: Q+ = [[1/4, -1/4], [-1/4, 1/4]]."""
        W = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        s = compute_bym2_scaling_factor(W)
        # marginal_vars = [1/4, 1/4]; geometric mean = 1/4
        assert abs(s - 0.25) < 1e-6


# ---------------------------------------------------------------------------
# _connect_islands
# ---------------------------------------------------------------------------

class TestConnectIslandsEdgeCases:
    def _mock_gdf(self, xs, ys):
        from unittest.mock import MagicMock
        gdf = MagicMock()
        gdf.geometry.centroid.x.values = np.array(xs)
        gdf.geometry.centroid.y.values = np.array(ys)
        return gdf

    def test_already_connected_returns_same(self):
        """If graph is already connected, _connect_islands should return it unchanged."""
        adj = build_grid_adjacency(3, 3)
        gdf = self._mock_gdf(
            xs=[float(c) for c in range(9)],
            ys=[0.0] * 9,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            adj_out = _connect_islands(adj, gdf)
        assert adj_out.n_components() == 1

    def test_four_isolated_nodes_all_connected(self):
        """4 isolated nodes at distinct positions should all merge into 1 component."""
        W = sparse.csr_matrix(np.zeros((4, 4), dtype=float))
        adj = AdjacencyMatrix(W=W, areas=["a", "b", "c", "d"])
        assert adj.n_components() == 4

        gdf = self._mock_gdf([0, 1, 2, 3], [0, 0, 0, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            adj_out = _connect_islands(adj, gdf)
        assert adj_out.n_components() == 1

    def test_island_connects_to_nearest_node(self):
        """Island should connect to the closer of two main-component nodes."""
        # Main chain: 0-1, island at x=10 (closer to node 1 at x=3 than node 0 at x=0)
        W = sparse.csr_matrix(np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=float))
        adj = AdjacencyMatrix(W=W, areas=["a", "b", "c"])
        gdf = self._mock_gdf([0.0, 3.0, 10.0], [0.0, 0.0, 0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            adj_out = _connect_islands(adj, gdf)

        # Node 2 (island) should be connected to node 1 (nearest)
        W_out = adj_out.to_dense()
        assert W_out[1, 2] == 1.0 or W_out[2, 1] == 1.0

    def test_warning_issued_for_island_connection(self):
        """_connect_islands must issue a UserWarning when it adds edges."""
        W = sparse.csr_matrix(np.zeros((3, 3), dtype=float))
        adj = AdjacencyMatrix(W=W, areas=["a", "b", "c"])
        gdf = self._mock_gdf([0, 5, 10], [0, 0, 0])

        with pytest.warns(UserWarning):
            _connect_islands(adj, gdf)
