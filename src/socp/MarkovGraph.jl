"""
    MarkovGraph

Module 0 primitive for the SOCP formulation (see `spec.txt`).

Stores an undirected graph together with a reversible Markov chain on it, using a
per-edge (rather than per-node-pair-matrix) representation: momenta, potential
gradients, and other edge quantities are each a single `Vector{Float64}` of length
`|E|`, with an arbitrary but fixed orientation per edge. Antisymmetry of edge fields
(Lemma 2.4 of Erbar et al.) is therefore hard-coded into the representation.

Fields:
- `n::Int`: number of vertices.
- `E::Vector{Tuple{Int,Int}}`: oriented edge list, `E[e] = (x, y)` meaning the edge
  is oriented from `x` to `y`.
- `π::Vector{Float64}`: stationary distribution.
- `Q::SparseMatrixCSC{Float64,Int}`: transition rate matrix.
- `κ::Vector{Float64}`: edge weights `κ[e] = Q[x,y] * π[x]` for `e = (x,y)`, which by
  reversibility equals `Q[y,x] * π[y]`.
- `D::SparseMatrixCSC{Float64,Int}`: `n × |E|` incidence matrix with `D*m == graph_divergence(G, m)`
  for any edge field `m`, cached at construction so repeated divergences (e.g. one per SOCP
  time step) don't rebuild it. `D[x,e] = -Q[x,y]`, `D[y,e] = Q[y,x]` for edge `e = (x,y)`.
"""
struct MarkovGraph
    n::Int
    E::Vector{Tuple{Int,Int}}
    π::Vector{Float64}
    Q::SparseMatrixCSC{Float64,Int}
    κ::Vector{Float64}
    D::SparseMatrixCSC{Float64,Int}
end

"""
    MarkovGraph(Q, π; rtol=1e-12) -> MarkovGraph

Construct a `MarkovGraph` from a transition matrix `Q` and stationary distribution
`π`. Fixes an arbitrary orientation for each undirected edge (the one with `i < j`
in the upper triangle of `Q`) and verifies reversibility `Q[x,y]π[x] == Q[y,x]π[y]`
to tolerance `rtol` (relative to `max(Q[x,y]π[x], Q[y,x]π[y])`).
"""
function MarkovGraph(Q::AbstractMatrix, π::AbstractVector; rtol::Float64=1e-12)
    n = size(Q, 1)
    @assert size(Q, 2) == n
    @assert length(π) == n

    Qs = sparse(Q)
    rows, cols, _ = findnz(Qs)
    E = Tuple{Int,Int}[]
    κ = Float64[]
    for (i, j) in zip(rows, cols)
        i < j || continue
        κ_ij = Qs[i, j] * π[i]
        κ_ji = Qs[j, i] * π[j]
        scale = max(abs(κ_ij), abs(κ_ji), 1e-300)
        @assert abs(κ_ij - κ_ji) / scale ≤ rtol "reversibility violated on edge ($i,$j): Q[i,j]π[i]=$κ_ij, Q[j,i]π[j]=$κ_ji"
        push!(E, (i, j))
        push!(κ, κ_ij)
    end

    D_I = Int[]; D_J = Int[]; D_V = Float64[]
    for (e, (x, y)) in enumerate(E)
        push!(D_I, x); push!(D_J, e); push!(D_V, -Qs[x, y])
        push!(D_I, y); push!(D_J, e); push!(D_V, Qs[y, x])
    end
    D = sparse(D_I, D_J, D_V, n, length(E))

    return MarkovGraph(n, E, collect(Float64, π), Qs, κ, D)
end

"""
    graph_gradient(G::MarkovGraph, φ::AbstractVector) -> Vector{Float64}

Compact-edge-vector counterpart of the dense [`graph_gradient`](@ref): `(∇φ)[e] = φ[x] - φ[y]`
for the oriented edge `e = (x, y)`.
"""
function graph_gradient(G::MarkovGraph, φ::AbstractVector)
    @assert length(φ) == G.n
    return [φ[x] - φ[y] for (x, y) in G.E]
end

"""
    graph_divergence(G::MarkovGraph, m::AbstractVector)

Compact-edge-vector counterpart of the dense [`graph_divergence`](@ref), specialized to a
single scalar per undirected edge (rather than a full antisymmetric `V×V` matrix). Agrees
with the dense version under `m_dense[x,y] = m[e], m_dense[y,x] = -m[e]`, and therefore
satisfies the same adjoint identity `⟨φ, div m⟩_π = -⟨∇φ, m⟩_Q` (with `⟨m,w⟩_Q := Σ_e κ[e] m[e] w[e]`).

Implemented as `G.D * m` (see the `D` field), so `m` may hold `Float64`s or, e.g., JuMP
variables/affine expressions when building an SOCP model.
"""
function graph_divergence(G::MarkovGraph, m::AbstractVector)
    @assert length(m) == length(G.E)
    return G.D * m
end
