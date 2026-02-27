module GraphTransportation

# dependencies
using SuiteSparse
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

# expose functionality for computing geodesics
export discrete_transport, transport_cost, action, barycenter, analysis

end
