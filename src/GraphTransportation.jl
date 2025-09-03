module GraphTransportation

using SuiteSparse
using SparseArrays
using LinearAlgebra
using BlockBandedMatrices
using CUDA
using ProgressMeter
using Base: signequal

include("utils.jl")
include("EarthMover.jl")
include("ErbarVector.jl")
include("galerkin/Chambolle.jl")
include("galerkin/ContinuityEnforcer.jl")
include("galerkin/ProximalAvgIndicator.jl")
include("galerkin/ProximalAction.jl")
include("galerkin/ProximalSignIndicator.jl")
include("galerkin/ProximalEqualityIndicator.jl")
include("galerkin/KProjection.jl")


export BBD

end
