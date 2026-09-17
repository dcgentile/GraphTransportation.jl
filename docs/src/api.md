# API Reference

## Module

```@docs
GraphTransportation
```

## Unified API

One function per task; `method=:socp` (default), `:shooting` or `:chambolle_pock`
selects the algorithm, and `barycenter` also accepts `method=:sinkhorn` (an entropic
barycenter for a ground cost, a different object; see its docstring). Keywords are
forwarded to the chosen implementation.

```@docs
geodesic
transport_cost
barycenter
analysis
GeodesicSolution
MarkovGraph
```

## SOCP implementation (internal)

```@docs
GraphTransportation.geodesic_socp
GraphTransportation.barycenter_socp
GraphTransportation.analyze_socp
```

## Hamiltonian shooting (exp/log maps)

```@docs
hamiltonian
hamiltonian_flow
integrate_hamiltonian
ρ_floor
PositivityFloorError
weighted_laplacian
solve_weighted_laplacian
momentum_to_potential
exp_map
log_map
log_map_mollified
GraphTransportation.analyze_shooting
```

## Chambolle-Pock implementation (reference, internal)

```@docs
GraphTransportation.discrete_transport
action
```

## Sinkhorn: entropic barycenters

```@docs
sinkhorn_barycenter
simplex_regression
ground_cost
graph_diameter
```

## Graph constructors

```@docs
markov_chain_from_edge_list
markov_chain_from_adjacency_matrix
markov_chain_from_weight_matrix
stationary_from_transition
triangle_markov_chain
square_markov_chain
cube_markov_chain
hypercube_markov_chain
weighted_hypercube_markov_chain
grid_markov_chain
triangular_prism_markov_chain
triangle_with_tail_markov_chain
T_markov_chain
double_T_markov_chain
ma_house_markov_chain
```

## Graph calculus

```@docs
graph_gradient
add_graph_gradient!
graph_divergence
graph_divergence!
laplacian_from_transition
metric_tensor
avg_operator
finite_difference_operator
```

## Admissible means

```@docs
geomean
logmean
logmean_partial_s
logmean_partial_t
```

## Chambolle-Pock data structures

```@docs
ErbarVector
ErbarCache
ErbarBundle
combine!
assign!
```
