"""
insurance-spatial: BYM2 spatial territory ratemaking and conformal prediction for UK personal lines.

Provides adjacency construction, BYM2 model fitting via PyMC, spatial diagnostics,
and territory relativity extraction - all in a pipeline designed to slot into an
existing GLM or GBM pricing workflow.

Also provides spatially weighted conformal prediction intervals via the
`insurance_spatial.conformal` sub-package:

    from insurance_spatial.conformal import SpatialConformalPredictor
"""

from insurance_spatial.adjacency import AdjacencyMatrix, build_grid_adjacency
from insurance_spatial.models import BYM2Model, BYM2Result
from insurance_spatial.diagnostics import MoranI, convergence_summary
from insurance_spatial.relativities import extract_relativities
from insurance_spatial import conformal  # noqa: F401 — expose sub-package

__all__ = [
    "AdjacencyMatrix",
    "build_grid_adjacency",
    "BYM2Model",
    "BYM2Result",
    "MoranI",
    "convergence_summary",
    "extract_relativities",
    "conformal",
    "__version__",
]

__version__ = "0.2.0"
