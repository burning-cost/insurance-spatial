"""
insurance-spatial: BYM2 spatial territory ratemaking and conformal prediction for UK personal lines.

Provides adjacency construction, BYM2 model fitting via PyMC or pyINLA,
spatial diagnostics, and territory relativity extraction - all in a pipeline
designed to slot into an existing GLM or GBM pricing workflow.

Two BYM2 backends are available:

- BYM2Model: PyMC MCMC sampler.  Exact posterior, full convergence diagnostics.
  Use for smaller grids (N < 500), prior sensitivity, and model development.

- BYM2InlaModel: pyINLA INLA approximation.  278x faster than MCMC on N=56.
  Use for large UK territory grids (N > 500) and iterative production reruns.
  Requires: uv add 'insurance-spatial[inla]'

Both backends produce results with the same territory_relativities() interface,
so downstream pricing code is backend-agnostic.

Also provides spatially weighted conformal prediction intervals via the
`insurance_spatial.conformal` sub-package:

    from insurance_spatial.conformal import SpatialConformalPredictor
"""

from insurance_spatial.adjacency import AdjacencyMatrix, build_grid_adjacency
from insurance_spatial.models import BYM2Model, BYM2Result
from insurance_spatial.diagnostics import MoranI, convergence_summary
from insurance_spatial.relativities import extract_relativities
from insurance_spatial.bym2_inla import BYM2InlaModel, BYM2InlaResult, InlaDiagnostics

__all__ = [
    "AdjacencyMatrix",
    "build_grid_adjacency",
    "BYM2Model",
    "BYM2Result",
    "BYM2InlaModel",
    "BYM2InlaResult",
    "InlaDiagnostics",
    "MoranI",
    "convergence_summary",
    "extract_relativities",
    "conformal",
    "__version__",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-spatial")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed


def __getattr__(name: str):
    if name == "conformal":
        from insurance_spatial import conformal as _conformal
        return _conformal
    raise AttributeError(f"module 'insurance_spatial' has no attribute {name!r}")
