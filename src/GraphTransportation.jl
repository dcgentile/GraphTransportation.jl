"""
    GraphTransportation

Julia package for Wasserstein geometry on graphs, implementing the framework of
Erbar, Rumpf, Schmitzer, and Simon — *Computation of optimal transport on
discrete metric measure spaces*.

The package provides:
- **Geodesics** (`discrete_transport`, `transport_cost`): compute optimal
  transport geodesics between probability measures on a graph via a
  Galerkin-discretised Chambolle-Pock primal-dual algorithm.
- **Barycenters** (`barycenter`, `iterated_barycenter`): compute Wasserstein
  Fréchet means via gradient descent on the graph Wasserstein space.
- **Coordinate recovery** (`analysis`): recover barycentric coordinates of a
  measure with respect to a reference family by solving a quadratic programme
  on the Gram matrix of logarithmic maps.
- **Entropic barycenters** (`sinkhorn_barycenter`, `simplex_regression`):
  entropy-regularised barycenter computation and coordinate recovery via the
  Sinkhorn algorithm.
"""
module GraphTransportation

# dependencies
using SuiteSparse
using Printf
using SparseArrays
using LinearAlgebra
using BlockBandedMatrices
using Convex, SCS
using ForwardDiff, Roots
using ProgressMeter
using Base: signequal

# include general helper functions
include("GraphCalculus.jl")
include("MarkovChains.jl")

# include components of Chambolle-Pock related functions
include("galerkin/ProximalAvgIndicator.jl")
include("galerkin/ProximalAction.jl")
include("galerkin/ProximalSignIndicator.jl")
include("galerkin/ContinuityEnforcer.jl")
include("galerkin/ProximalEqualityIndicator.jl")
include("galerkin/KProjection.jl")

# include the abstraction for the vector space defined in Erbar et al 2020
include("ErbarVector.jl")

# some tests to help us stay sane!
include("tests/Inclusion.jl")

# include the Chambolle-Pock routine
include("galerkin/Chambolle.jl")

# include functionality for computing geodesics
include("EarthMover.jl")

# include functionality for barycenter synthesis
include("Barycenters.jl")
include("Sinkhorn.jl")

# expose functionality for computing geodesics
export discrete_transport, transport_cost, action, barycenter, iterated_barycenter, analysis
export sinkhorn_barycenter, simplex_regression

end
