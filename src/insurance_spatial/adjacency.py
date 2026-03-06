"""
Adjacency matrix construction for spatial territory models.

Provides two paths:

1. From a GeoDataFrame (or GeoJSON file) of polygon boundaries - uses Queen
   contiguity (shared edge or vertex) via libpysal. This is the preferred route
   when you have actual boundary polygons.

2. From a simple grid - useful for synthetic data and testing.

The output is an AdjacencyMatrix dataclass that carries the N×N dense (for small
grids) or scipy sparse adjacency matrix W, the ordered list of area identifiers,
and the BYM2 scaling factor.  W is binary and symmetric.

Island handling
---------------
The ICAR model requires a connected graph. Any disconnected components (islands,
isolated areas) must be joined to the main graph before fitting. We do this by
connecting each island node to its nearest node in the largest connected component
using Euclidean distance between centroids. This is a library-level policy choice —
document it in your model documentation.

Scaling factor
--------------
The BYM2 scaling factor s = geometric_mean(diag(Q+)) where Q+ is the Moore-Penrose
pseudoinverse of the ICAR precision matrix Q = D - W.  We compute this via
eigendecomposition of Q (dense, for moderate N up to ~3,000) or via a sparse
Cholesky with sum-to-zero constraint (for larger N).

The ICAR precision matrix has exactly one zero eigenvalue corresponding to the
constant eigenvector.  We zero this eigenvalue before inverting to obtain Q+.

Cache the result: the scaling factor depends only on the graph topology, not the
data, so you only need to recompute it when the adjacency structure changes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass
class AdjacencyMatrix:
    """
    Container for an adjacency matrix and associated metadata.

    Attributes
    ----------
    W :
        N×N binary symmetric adjacency matrix as a scipy CSR sparse matrix.
        W[i, j] = 1 if areas i and j are neighbours, 0 otherwise.
        No self-loops: diagonal is zero.
    areas :
        Ordered sequence of area identifiers (postcode sectors, district codes, etc.).
        Position i in this list corresponds to row/column i of W.
    scaling_factor :
        BYM2 scaling factor s for this graph.  Computed lazily on first access.
    """

    W: sparse.csr_matrix
    areas: list[str]
    _scaling_factor: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        n = len(self.areas)
        if self.W.shape != (n, n):
            raise ValueError(
                f"W has shape {self.W.shape} but {n} area labels were provided"
            )
        if not isinstance(self.W, sparse.csr_matrix):
            self.W = sparse.csr_matrix(self.W)

    @property
    def n(self) -> int:
        """Number of areas."""
        return len(self.areas)

    @property
    def scaling_factor(self) -> float:
        """BYM2 scaling factor, computed on first access and cached."""
        if self._scaling_factor is None:
            self._scaling_factor = compute_bym2_scaling_factor(self.W)
        return self._scaling_factor

    def to_dense(self) -> np.ndarray:
        """Return W as a dense N×N float64 array."""
        return self.W.toarray().astype(np.float64)

    def to_edge_list(self) -> np.ndarray:
        """
        Return edges as an (E, 2) int array of (i, j) pairs where i < j.
        Suitable for use as the ``edge_list`` argument of some spatial libraries.
        """
        cx = self.W.tocoo()
        mask = cx.row < cx.col
        return np.stack([cx.row[mask], cx.col[mask]], axis=1)

    def n_components(self) -> int:
        """Number of connected components.  Should be 1 for a valid ICAR model."""
        n, _ = connected_components(self.W, directed=False)
        return n

    def neighbour_counts(self) -> np.ndarray:
        """Return array of length N with number of neighbours per area."""
        return np.asarray(self.W.sum(axis=1)).ravel().astype(int)

    def area_index(self) -> dict[str, int]:
        """Return mapping from area identifier to matrix row/column index."""
        return {a: i for i, a in enumerate(self.areas)}


def compute_bym2_scaling_factor(W: sparse.spmatrix) -> float:
    """
    Compute the BYM2 scaling factor for a given adjacency matrix.

    s = geometric_mean(diag(Q+))

    where Q = D - W is the ICAR precision matrix (Laplacian) and Q+ is its
    Moore-Penrose pseudoinverse.  Q has rank N-1 with a zero eigenvalue
    corresponding to the constant eigenvector; Q+ sets this eigenvalue to
    zero in the inverse (rather than 1/0).

    Implementation: dense eigendecomposition of Q.  This is O(N^2) in memory
    and O(N^3) in time - acceptable for N ≤ ~3,000 (postcode districts).  For
    N ≈ 11,200 (postcode sectors), run this once offline, cache the result, and
    pass it directly to AdjacencyMatrix(_scaling_factor=cached_value).

    For a connected graph, the scaling factor s lies in (0, ∞) and represents
    the geometric mean of the ICAR marginal variances.  Dividing the ICAR
    samples by sqrt(s) normalises them to unit marginal variance, which is the
    key step in the BYM2 parameterisation.

    Parameters
    ----------
    W :
        N×N binary symmetric adjacency matrix (scipy sparse).

    Returns
    -------
    float
        Scaling factor s > 0.
    """
    W_arr = sparse.csr_matrix(W, dtype=np.float64).toarray()
    N = W_arr.shape[0]
    d = W_arr.sum(axis=1)
    Q = np.diag(d) - W_arr  # N×N Laplacian matrix

    # Eigendecomposition: Q = V diag(lambda) V^T
    # Q is symmetric PSD with exactly one zero eigenvalue for a connected graph
    eigenvalues, eigenvectors = np.linalg.eigh(Q)

    # Zero out eigenvalues below a relative threshold (the null-space eigenvalue)
    # Use relative threshold: eigenvalues below max_eigenvalue * tol are zeroed
    tol = 1e-10 * eigenvalues[-1]  # eigenvalues sorted ascending by eigh
    inv_eigenvalues = np.where(eigenvalues > tol, 1.0 / eigenvalues, 0.0)

    # Q+ = V diag(1/lambda+) V^T  (pseudoinverse)
    # diag(Q+)[i] = sum_k (V[i,k]^2 * inv_eigenvalues[k])
    marginal_vars = (eigenvectors ** 2) @ inv_eigenvalues  # shape (N,)

    scaling_factor = float(np.exp(np.mean(np.log(marginal_vars))))
    return scaling_factor


def build_grid_adjacency(
    nrows: int,
    ncols: int,
    connectivity: str = "rook",
) -> AdjacencyMatrix:
    """
    Build an adjacency matrix for a regular grid of nrows × ncols cells.

    Useful for testing and for demonstrating the library on synthetic data.

    Parameters
    ----------
    nrows, ncols :
        Grid dimensions.
    connectivity :
        ``"rook"`` - shared edge only (4-connectivity).
        ``"queen"`` - shared edge or vertex (8-connectivity).

    Returns
    -------
    AdjacencyMatrix
        W is (nrows*ncols) × (nrows*ncols), areas are labelled ``"r{i}c{j}"``.
    """
    N = nrows * ncols
    rows_idx = []
    cols_idx = []

    def cell(r: int, c: int) -> int:
        return r * ncols + c

    for r in range(nrows):
        for c in range(ncols):
            idx = cell(r, c)
            # Rook neighbours
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    rows_idx.append(idx)
                    cols_idx.append(cell(nr, nc))
            # Queen extra diagonal neighbours
            if connectivity == "queen":
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < nrows and 0 <= nc < ncols:
                        rows_idx.append(idx)
                        cols_idx.append(cell(nr, nc))

    data = np.ones(len(rows_idx), dtype=np.float64)
    W = sparse.csr_matrix((data, (rows_idx, cols_idx)), shape=(N, N))
    areas = [f"r{r}c{c}" for r in range(nrows) for c in range(ncols)]
    return AdjacencyMatrix(W=W, areas=areas)


def from_geojson(
    path: str,
    area_col: str,
    connectivity: str = "queen",
    fix_islands: bool = True,
) -> AdjacencyMatrix:
    """
    Build an adjacency matrix from a GeoJSON or Shapefile of polygons.

    Requires the optional ``geo`` dependencies: ``geopandas`` and ``libpysal``.

    Parameters
    ----------
    path :
        Path to a GeoJSON, Shapefile, or any format readable by geopandas.
    area_col :
        Column in the file that contains area identifiers (e.g. postcode sector codes).
    connectivity :
        ``"queen"`` (default) or ``"rook"`` or ``"knn"`` (k=6 nearest centroids).
        Queen contiguity is standard for UK territory models.
    fix_islands :
        If True, connect any disconnected graph components to the nearest node in
        the main component by adding edges.  Required for ICAR model validity.

    Returns
    -------
    AdjacencyMatrix
    """
    try:
        import geopandas as gpd  # noqa: F401
        from libpysal.weights import Queen, Rook, KNN
    except ImportError as exc:
        raise ImportError(
            "from_geojson() requires geopandas and libpysal. "
            "Install with: pip install insurance-spatial[geo]"
        ) from exc

    gdf = gpd.read_file(path)
    if area_col not in gdf.columns:
        raise ValueError(f"Column '{area_col}' not found. Available: {list(gdf.columns)}")

    gdf = gdf.reset_index(drop=True)
    areas = gdf[area_col].astype(str).tolist()

    if connectivity == "queen":
        w = Queen.from_dataframe(gdf, silence_warnings=True)
    elif connectivity == "rook":
        w = Rook.from_dataframe(gdf, silence_warnings=True)
    elif connectivity == "knn":
        w = KNN.from_dataframe(gdf, k=6, silence_warnings=True)
    else:
        raise ValueError(f"Unknown connectivity '{connectivity}'. Use 'queen', 'rook', or 'knn'.")

    # Build symmetric binary adjacency matrix from libpysal weight object
    N = len(areas)
    rows_idx: list[int] = []
    cols_idx: list[int] = []
    for i, neighbours in w.neighbors.items():
        for j in neighbours:
            rows_idx.append(i)
            cols_idx.append(j)

    data = np.ones(len(rows_idx), dtype=np.float64)
    W = sparse.csr_matrix((data, (rows_idx, cols_idx)), shape=(N, N))

    adj = AdjacencyMatrix(W=W, areas=areas)

    if fix_islands and adj.n_components() > 1:
        adj = _connect_islands(adj, gdf)

    return adj


def _connect_islands(adj: AdjacencyMatrix, gdf) -> AdjacencyMatrix:  # type: ignore[no-untyped-def]
    """
    Connect disconnected components by linking island nodes to their nearest
    node in the largest connected component.

    Nearest is measured by Euclidean distance between polygon centroids.
    """
    n_comp, labels = connected_components(adj.W, directed=False)
    if n_comp == 1:
        return adj

    # Find the largest component
    counts = np.bincount(labels)
    main_label = int(np.argmax(counts))

    centroids_x = gdf.geometry.centroid.x.values
    centroids_y = gdf.geometry.centroid.y.values

    W_lil = adj.W.tolil()
    n_islands_connected = 0

    for comp_label in range(n_comp):
        if comp_label == main_label:
            continue
        island_nodes = np.where(labels == comp_label)[0]
        main_nodes = np.where(labels == main_label)[0]

        for island_node in island_nodes:
            ix, iy = centroids_x[island_node], centroids_y[island_node]
            mx = centroids_x[main_nodes]
            my = centroids_y[main_nodes]
            dists = np.sqrt((mx - ix) ** 2 + (my - iy) ** 2)
            nearest_main = main_nodes[int(np.argmin(dists))]
            W_lil[island_node, nearest_main] = 1.0
            W_lil[nearest_main, island_node] = 1.0
            n_islands_connected += 1

    warnings.warn(
        f"Graph had {n_comp} connected components. "
        f"Connected {n_islands_connected} island nodes to nearest mainland node.",
        UserWarning,
        stacklevel=3,
    )
    W_new = W_lil.tocsr()
    return AdjacencyMatrix(W=W_new, areas=adj.areas)
