"""
    analyze_socp(G, target, refs; N=10, optimizer=Clarabel.Optimizer,
                 compute_condition=false, return_system=false) -> λ̂ (or (λ̂, A))

Module 4's `:socp` backend (`spec.txt`): recover the barycentric coordinates of
`target` with respect to `refs` by computing, for each reference, the initial
momentum of the geodesic from `target` to it via `geodesic_socp` (Module 1) rather
than `discrete_transport`, then solving the exact same Gram-matrix simplex QP as the
existing `analysis` (see `solve_barycentric_coordinates_qp`) — reusing that
already-validated formulation rather than re-deriving spec.txt's potential-based Gram
matrix (`A_ij = Σ_e θ(target)_e (∇φ_i)_e (∇φ_j)_e κ_e`) from scratch. The two are
equivalent: `spec.txt` notes `m = θ(target)∘∇φ`, so `(∇φ_i)_e = m_{i,e}(0)/θ(target)_e`,
and both `discrete_transport` and `geodesic_socp` produce the same `m` (Module 1's
§1.3.2 gate).

`refs` is a vector of reference probability densities on `G`. Returns the recovered
weight vector `λ̂` (from `Convex.jl`/SCS on the small `p × p` Gram matrix), or
`(λ̂, A)` if `return_system=true`.
"""
function analyze_socp(G::MarkovGraph, target::AbstractVector, refs::Vector{<:AbstractVector};
                       N::Int=10, optimizer=Clarabel.Optimizer,
                       compute_condition::Bool=false, return_system::Bool=false)
    tangent_vectors = map(refs) do ref
        m0 = geodesic_socp(G, target, ref; N=N, optimizer=optimizer).m0
        m_dense = zeros(G.n, G.n)
        for (e, (x, y)) in enumerate(G.E)
            m_dense[x, y] = m0[e]
            m_dense[y, x] = -m0[e]
        end
        m_dense
    end
    g = metric_tensor(target)

    return solve_barycentric_coordinates_qp(tangent_vectors, g;
                                             compute_condition=compute_condition, return_system=return_system)
end
