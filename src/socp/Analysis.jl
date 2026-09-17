"""
    analyze_socp(G, target, refs; N=10, optimizer=Clarabel.Optimizer,
                 convention=:potential, compute_condition=false, return_system=false)
        -> λ̂ (or (λ̂, A))

The `:socp` analysis backend: recover the barycentric coordinates of
`target` with respect to `refs` by solving, for each reference, the geodesic SOCP
from `target` to it (`geodesic_socp`), building a `p × p` Gram matrix of
the resulting tangent vectors at `target`, and solving the simplex QP
`min_{λ∈Δ} λᵀAλ` shared with the existing `analysis` (`solve_barycentric_coordinates_qp`).

`convention` selects which tangent-vector representation the Gram matrix is built from:

- `:potential` (default): the Riemannian Gram matrix at `target`,
  `A_ij = Σ_e κ_e θ(target)_e (∇φ_i)_e (∇φ_j)_e`, with `φ_i` the endpoint potential
  `GeodesicSolution.φ0` — the exact gradient of `𝒲_h²(target, ref_i)` with respect to
  `target`. Since a `barycenter_socp` solution is, by its KKT conditions, exactly a
  point where `Σᵢ λᵢ φ_i = const` on its support, this recovers the synthesis weights
  of an SOCP barycenter to solver tolerance at *any* `N`, including `N=2`.
- `:momentum`: the earlier implementation, which used the initial momentum `m0` as the
  tangent vector with the dense `metric_tensor(target)` weighting, exactly as the
  Chambolle-Pock-based `analysis` does. `m0` lives on the first time *interval*
  (`m0 = -(1/2h) θ(ρ̄_{1/2}) ∇ψ_{1/2}`, with the midpoint density and the half-step
  potential), so it is only an `O(h)` proxy for the endpoint potential, and
  `Σᵢ λᵢ m0ᵢ ≈ 0` is *not* the stationarity condition the SOCP barycenter satisfies.
  At coarse `N` this mismatch can be large (tens of percent recovery error at `N=2` on
  irregular graphs). Kept for comparison experiments only.

`refs` is a vector of reference probability densities on `G`. Returns the recovered
weight vector `λ̂` (from `Convex.jl`/SCS on the small `p × p` Gram matrix), or
`(λ̂, A)` if `return_system=true`.
"""
function analyze_socp(G::MarkovGraph, target::AbstractVector, refs::Vector{<:AbstractVector};
                       N::Int=10, optimizer=Clarabel.Optimizer, convention::Symbol=:potential,
                       compute_condition::Bool=false, return_system::Bool=false)
    convention in (:potential, :momentum) ||
        throw(ArgumentError("convention must be :potential or :momentum, got $convention"))

    geodesics = [geodesic_socp(G, target, ref; N=N, optimizer=optimizer) for ref in refs]

    if convention == :potential
        return potential_gram_qp(G, target, [geo.φ0 for geo in geodesics];
                                 compute_condition=compute_condition, return_system=return_system)
    else
        tangent_vectors = map(geodesics) do geo
            m_dense = zeros(G.n, G.n)
            for (e, (x, y)) in enumerate(G.E)
                m_dense[x, y] = geo.m0[e]
                m_dense[y, x] = -geo.m0[e]
            end
            m_dense
        end
        g = metric_tensor(target)
    end

    return solve_barycentric_coordinates_qp(tangent_vectors, g;
                                             compute_condition=compute_condition, return_system=return_system)
end

"""
    potential_gram_qp(G, target, potentials; compute_condition=false, return_system=false) -> λ̂ (or (λ̂, A))

Gram matrix and simplex QP, shared by the `:socp` and `:shooting` backends: given one potential
`φ_i` per reference (the geodesic from `target` to `ref_i`, in any sign/scale convention
common to all `i`), assemble the Gram matrix
`A_ij = Σ_e κ_e θ(target)_e (∇φ_i)_e (∇φ_j)_e` and solve `min_{λ∈Δ} λᵀAλ` via
`solve_barycentric_coordinates_qp`.
"""
function potential_gram_qp(G::MarkovGraph, target::AbstractVector, potentials;
                           compute_condition::Bool=false, return_system::Bool=false)
    tangent_vectors = [graph_gradient(G, φ) for φ in potentials]
    g = G.κ .* metric_tensor(G, target)
    return solve_barycentric_coordinates_qp(tangent_vectors, g;
                                             compute_condition=compute_condition, return_system=return_system)
end
