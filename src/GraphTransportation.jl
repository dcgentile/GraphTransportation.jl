module GraphTransportation

# dependencies
using SuiteSparse
using SparseArrays
using LinearAlgebra
using BlockBandedMatrices
using CUDA
using ProgressMeter
using Base: signequal

# include general helper functions
include("utils.jl")

# include components of Chambolle-Pock related functions
include("galerkin/ContinuityEnforcer.jl")
include("galerkin/ProximalAvgIndicator.jl")
include("galerkin/ProximalAction.jl")
include("galerkin/ProximalSignIndicator.jl")
include("galerkin/ProximalEqualityIndicator.jl")
include("galerkin/KProjection.jl")

# include the Chambolle-Pock routine
include("galerkin/Chambolle.jl")

# include functionality for computing geodesics
include("ErbarVector.jl")
include("EarthMover.jl")


# expose functionality for computing geodesics
export BBD

end
