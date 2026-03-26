# API Reference

## Module

```@docs
GraphTransportation
```

## Geodesics

```@docs
discrete_transport
transport_cost
action
```

## Barycenters

```@docs
barycenter
iterated_barycenter
analysis
```

## Entropic barycenters

```@docs
sinkhorn_barycenter
simplex_regression
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

## Data structures

```@docs
ErbarVector
ErbarCache
ErbarBundle
combine!
assign!
```
