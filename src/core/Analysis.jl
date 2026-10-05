# Shared analysis machinery: the simplex QP on a Gram matrix of tangent vectors, used
# by every analysis backend (Chambolle-Pock `analysis`, `analyze_socp`, `analyze_shooting`).

"""
    solve_barycentric_coordinates_qp(tangent_vectors, g; compute_condition=false, return_system=false)

Shared Gram-matrix-assembly and simplex-QP-solve core of `analysis`: given the initial
tangent vectors (dense `V × V` antisymmetric matrices, one per reference) of the
geodesics from a target measure to each reference, and the target's metric tensor `g`,
assembles `A[i,j] = Σ_{x,y} tangent_vectors[i][x,y] * tangent_vectors[j][x,y] * g[x,y]`
and solves `min_{w≥0, Σw=1} w'Aw` exactly (`solve_simplex_qp`). Factored out of `analysis` so
`analyze_socp` (which sources tangent vectors from `geodesic_socp` instead of
`discrete_transport`) can reuse the exact same, already-validated Gram/QP formulation
rather than re-deriving it. Returns `λ̂` as a `Vector` (or `(λ̂, A)` with
`return_system=true`).
"""
function solve_barycentric_coordinates_qp(tangent_vectors, g; compute_condition=false, return_system=false)
    p = length(tangent_vectors)
    A = zeros(p, p)
    for i=1:p, j=i:p
        A[i,j] = A[j,i] = sum(tangent_vectors[i] .* tangent_vectors[j] .* g)
    end

    if compute_condition
        e = abs.(eigvals(A))
        κ = maximum(e) / minimum(e)
        println("Estimated condition number of analysis matrix: $(κ)")
    end
    λ = solve_simplex_qp(A)
    return return_system ? (λ, A) : λ
end

# Largest p for which `solve_simplex_qp` enumerates all 2^p - 1 supports.
const SIMPLEX_QP_ENUM_MAX = 12

"""
    solve_simplex_qp(A) -> λ

Minimize `λᵀAλ` over the probability simplex for a symmetric positive semidefinite `A`.
For small `p` this is exact to rounding: the minimizer has some support `S` on which it
solves the KKT system `A_SS x = c·1, Σx = 1` (solved as the bordered system
`[A_SS 1; 1ᵀ 0]`, which stays nonsingular when `A_SS` is singular with a null vector off
the sum-zero hyperplane — exactly the case `c = 0` of an exactly recovered barycenter, where
`A_SS⁻¹1` does not exist). Every support is tried, candidates with a negative entry are
dropped, and the best objective wins (`2^p - 1` small solves). Beyond
`SIMPLEX_QP_ENUM_MAX` references, or if no support yields a candidate (singular `A`), falls
back to Clarabel at its default (~1e-8) tolerance.
"""
function solve_simplex_qp(A::AbstractMatrix)
    p = size(A, 1)
    p <= SIMPLEX_QP_ENUM_MAX || return _simplex_qp_conic(A)
    best, λbest = Inf, nothing
    for mask in 1:(2^p - 1)
        S = [i for i in 1:p if (mask >> (i - 1)) & 1 == 1]
        k = length(S)
        K = [A[S, S] ones(k); ones(1, k) 0.0]
        sol = try K \ [zeros(k); 1.0] catch; continue end
        all(isfinite, sol) || continue
        x = sol[1:k]
        all(>=(0), x) || continue
        λ = zeros(p); λ[S] = x
        f = dot(λ, A * λ)
        f < best && ((best, λbest) = (f, λ))
    end
    λbest === nothing ? _simplex_qp_conic(A) : λbest
end

function _simplex_qp_conic(A)
    n = size(A, 1)
    # the minimizer is invariant under A -> cA, and the solver stops on absolute tolerances
    A = A ./ max(maximum(abs, A), floatmin())
    x = Variable(n)
    problem = minimize(quadform(x, Symmetric(A)), [x >= 0, sum(x) == 1])
    Convex.solve!(problem, Clarabel.Optimizer; silent=true)
    return vec(x.value)
end

"""
    potential_gram_qp(G, target, potentials; compute_condition=false, return_system=false) -> λ̂ (or (λ̂, A))

Gram matrix and simplex QP, shared by the `:socp` and `:shooting` backends: given one potential
`φ_i` per reference (the geodesic from `target` to `ref_i`, in any sign/scale convention
common to all `i`), assemble the Gram matrix
`A_ij = Σ_e κ_e θ(target)_e (∇φ_i)_e (∇φ_j)_e` (with `θ = G.mean`) and solve `min_{λ∈Δ} λᵀAλ` via
`solve_barycentric_coordinates_qp`.
"""
function potential_gram_qp(G::MarkovGraph, target::AbstractVector, potentials;
                           compute_condition::Bool=false, return_system::Bool=false)
    tangent_vectors = [graph_gradient(G, φ) for φ in potentials]
    g = G.κ .* metric_tensor(G, target)          # θ = G.mean
    return solve_barycentric_coordinates_qp(tangent_vectors, g;
                                             compute_condition=compute_condition, return_system=return_system)
end
