# GraphTransportation.jl

[![CI](https://github.com/dcgentile/GraphTransportation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/dcgentile/GraphTransportation.jl/actions/workflows/CI.yml)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://dcgentile.github.io/GraphTransportation.jl/dev/)
[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://dcgentile.github.io/GraphTransportation.jl/stable/)

A Julia package for discrete transport geometry on graphs, implementing the framework of
Erbar, Rumpf, Schmitzer, and Simon —
*Computation of optimal transport on discrete metric measure spaces*.

Full documentation is available at **https://dcgentile.github.io/GraphTransportation.jl**.

## Installation

```julia
]add GraphTransportation
```

## Quick start

```julia
using GraphTransportation

# Two-point graph with its stationary distribution
Q = [0.0 1.0; 1.0 0.0]
π = [0.5, 0.5]
G = MarkovGraph(Q, π)

# Two Dirac masses (densities with respect to π)
a = [2.0, 0.0]
b = [0.0, 2.0]

# Discrete transport geodesic and distance (method=:socp by default;
# :shooting for strictly positive measures, :chambolle_pock for the paper's solver)
geo  = geodesic(G, a, b; N=20)
dist = transport_cost(G, a, b; N=20)

# Discrete transport barycenter with weights (0.75, 0.25)
bary, J, _ = barycenter(G, [a, b], [0.75, 0.25]; N=20)

# Recover barycentric coordinates
coords = analysis(G, bary, [a, b]; N=20)
```
