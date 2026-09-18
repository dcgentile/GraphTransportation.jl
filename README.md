# GraphTransportation.jl

[![CI](https://github.com/dcgentile/GraphTransportation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/dcgentile/GraphTransportation.jl/actions/workflows/CI.yml)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://dcgentile.github.io/GraphTransportation.jl/dev/)
[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://dcgentile.github.io/GraphTransportation.jl/stable/)

Optimal transport geometry for probability measures on graphs, in Julia.

A graph is given as a reversible Markov chain (transition matrix `Q`, stationary
distribution `π`). The package equips the measures on it with the discrete transport
metric of Maas and of Chow, Huang, Li and Zhou, for any admissible mean (geometric,
arithmetic, harmonic, logarithmic), and computes:

- **geodesics and distances** between measures, either as one second-order cone
  program (any measures, `O(1/N)` in time) or by Hamiltonian shooting (strictly
  positive measures, exact in time, with `exp_map` / `log_map`);
- **barycenters**, as one joint convex program solved to its global optimum, or by
  Riemannian gradient descent along shooting geodesics;
- **barycentric coordinates** of a measure with respect to reference measures, from the
  Gram matrix of the geodesics' initial potentials (the barycentric coding model);
- **entropic transport** for a ground cost on the nodes (Sinkhorn barycenters and
  barycentric coordinates), for comparison.

One function per task, with `method=:socp` (default), `:shooting`, `:chambolle_pock`
(the Galerkin primal-dual scheme of Erbar, Rumpf, Schmitzer and Simon, kept as a
reference implementation) or `:sinkhorn`.

Documentation, with an executed tutorial and a references page:
**https://dcgentile.github.io/GraphTransportation.jl**.

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
G = MarkovGraph(Q, π)                 # mean=GeometricMean() by default

# Two Dirac masses (densities with respect to π)
a = [2.0, 0.0]
b = [0.0, 2.0]

# Discrete transport geodesic and squared distance (method=:socp by default;
# :shooting for strictly positive measures)
geo = geodesic(G, a, b; N=20)
W2  = transport_cost(G, a, b; N=20)

# Discrete transport barycenter with weights (0.75, 0.25)
bary, J, _ = barycenter(G, [a, b], [0.75, 0.25]; N=20)

# Recover the barycentric coordinates
coords = analysis(G, bary, [a, b]; N=20)

# A different transport metric: the same graph with the harmonic mean
transport_cost(MarkovGraph(G; mean=HarmonicMean()), a, b; N=20)
```

## Citing

The package accompanies

> D. Gentile, J. M. Murphy. *Static and dynamic approaches to computing barycenters of
> probability measures on graphs.* arXiv:2603.26940, 2026.
> [https://arxiv.org/abs/2603.26940](https://arxiv.org/abs/2603.26940)

The literature behind each part of the package (the discrete transport metric, its
computation, barycenters and coordinates, entropic transport) is collected on the
documentation's References page.
