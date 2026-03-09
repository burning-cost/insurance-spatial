"""
insurance-spatial: BYM2 spatial territory ratemaking for UK personal lines.

Provides adjacency construction, BYM2 model fitting via PyMC, spatial diagnostics,
and territory relativity extraction - all in a pipeline designed to slot into an
existing GLM or GBM pricing workflow.
"""

from insurance_spatial.adjacency import AdjacencyMatrix, build_grid_adjacency
from insurance_spatial.models import BYM2Model, BYM2Result
from insurance_spatial.diagnostics import MoranI, convergence_summary
from insurance_spatial.relativities import extract_relativities

__all__ = [
    "AdjacencyMatrix",
    "build_grid_adjacency",
    "BYM2Model",
    "BYM2Result",
    "MoranI",
    "convergence_summary",
    "extract_relativities",
    "__version__",
]

__version__ = "0.1.0"
