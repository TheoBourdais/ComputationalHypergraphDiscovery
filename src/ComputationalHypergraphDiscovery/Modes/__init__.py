"""ComputationalHypergraphDiscovery.Modes package

This package implements the different functions needed to manipulate the modes, 
kernels, and kernel matrices for computational hypergraph discovery.

Submodules:
- container: Handles storage and manipulation of modes and their kernel matrices. 
- kernels: Provides algorithms representing a kernel for a mode in a hypergraph.

For more detailed information on each submodule, use help() on the respective submodule."""

from .kernels import (
    ModeKernelList,
    ModeKernel,
    LinearMode,
    QuadraticMode,
    GaussianMode,
    SklearnMode,
    PreComputedMode,
)
from .container import ModeContainer
