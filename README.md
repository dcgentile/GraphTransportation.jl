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

# Two-point graph
Q = [0.0 1.0; 1.0 0.0]

# Two Dirac masses
a = [2.0, 0.0]
b = [0.0, 2.0]

# Discrete transport geodesic and transport cost
geo  = discrete_transport(Q, a, b)
dist = transport_cost(Q, a, b)

# Discrete transport barycenter with weights (0.75, 0.25)
M    = hcat(a, b)
bary = barycenter(M, [0.75, 0.25], Q)

# Recover barycentric coordinates
coords = analysis(bary, M, Q)
```
