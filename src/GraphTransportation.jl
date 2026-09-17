"""
    GraphTransportation

Julia package for discrete transport geometry on graphs, implementing the framework of
Erbar, Rumpf, Schmitzer, and Simon — *Computation of optimal transport on
discrete metric measure spaces*.

The package provides:
- **Geodesics** (`discrete_transport`, `transport_cost`): compute discrete
  transport geodesics between probability measures on a graph via a
  Galerkin-discretised Chambolle-Pock primal-dual algorithm.
- **Barycenters** (`barycenter`): compute discrete transport Fréchet means via
  gradient descent.
- **Coordinate recovery** (`analysis`): recover barycentric coordinates of a
  measure with respect to a reference family by solving a quadratic programme
  on the Gram matrix of logarithmic maps.
- **Entropic optimal transport barycenters** (`sinkhorn_barycenter`, `simplex_regression`):
  entropic optimal transport barycenter computation and coordinate recovery via
  the Sinkhorn algorithm.
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
using JuMP, Clarabel
using Optim

# include general helper functions
# --- core: graph calculus, Markov chains, graph constructors, the compact MarkovGraph
#     representation, and the analysis QP shared by all three methods
include("core/GraphCalculus.jl")
include("core/Means.jl")
include("core/MarkovChains.jl")
include("core/CommonGraphs.jl")
include("core/MarkovGraph.jl")
include("core/Analysis.jl")

# --- Chambolle-Pock: the paper's reference implementation (Galerkin-discretised
#     primal-dual geodesics, gradient-descent barycenters, momentum-based analysis)
include("chambolle_pock/ProximalAvgIndicator.jl")
include("chambolle_pock/ProximalAction.jl")
include("chambolle_pock/ProximalSignIndicator.jl")
include("chambolle_pock/ContinuityEnforcer.jl")
include("chambolle_pock/ProximalEqualityIndicator.jl")
include("chambolle_pock/KProjection.jl")
include("chambolle_pock/ErbarVector.jl")
include("chambolle_pock/Chambolle.jl")
include("chambolle_pock/Geodesic.jl")
include("chambolle_pock/Barycenter.jl")

# --- SOCP: geodesics and barycenters as second-order-cone programs; potential-based analysis
include("socp/Geodesic.jl")
include("socp/Barycenter.jl")
include("socp/Analysis.jl")

# --- Hamiltonian shooting: exp/log maps and the :shooting analysis backend
include("shooting/Hamiltonian.jl")
include("shooting/ExpLog.jl")

# --- Sinkhorn: entropic barycenters and simplex-regression analysis
include("sinkhorn/Sinkhorn.jl")

# --- unified entry points (geodesic / transport_cost / barycenter / analysis; method=)
include("API.jl")

# core API
# unified API: one function per task, `method=:socp | :shooting | :chambolle_pock`
export geodesic, transport_cost, barycenter, analysis, GeodesicSolution

# Chambolle-Pock (reference implementation); discrete_transport is not exported
export action

# Sinkhorn (entropic, ground-cost based; also barycenter(...; method=:sinkhorn))
export sinkhorn_barycenter, simplex_regression, ground_cost, graph_diameter

# Markov chain constructors
export markov_chain_from_edge_list, markov_chain_from_adjacency_matrix
export markov_chain_from_weight_matrix, stationary_from_transition

# predefined graphs
export triangle_markov_chain, triangle_with_tail_markov_chain
export square_markov_chain, T_markov_chain, double_T_markov_chain
export triangular_prism_markov_chain, cube_markov_chain
export hypercube_markov_chain, weighted_hypercube_markov_chain
export grid_markov_chain, ma_house_markov_chain

# graph calculus
export graph_gradient, add_graph_gradient!, graph_divergence, graph_divergence!
export laplacian_from_transition, metric_tensor, avg_operator, finite_difference_operator

# SOCP formulation: graph primitives, geodesics, barycenters, analysis
export MarkovGraph   # geodesic_socp / barycenter_socp / analyze_socp are internal (use method=:socp)

# Hamiltonian shooting: exp/log maps and the :shooting analysis backend
export hamiltonian, hamiltonian_flow, integrate_hamiltonian, ρ_floor, PositivityFloorError
export weighted_laplacian, solve_weighted_laplacian, momentum_to_potential, exp_map, log_map, log_map_mollified   # analyze_shooting is internal (use method=:shooting)

# admissible means
export geomean, logmean, logmean_partial_s, logmean_partial_t
export AdmissibleMean, GeometricMean, ArithmeticMean, HarmonicMean, LogarithmicMean, QuadLogMean
export partial_s, partial_t

# data structures
export ErbarVector, ErbarCache, ErbarBundle, combine!, assign!

end
