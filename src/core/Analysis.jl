# Shared analysis machinery: the simplex QP on a Gram matrix of tangent vectors, used
# by every analysis backend (Chambolle-Pock `analysis`, `analyze_socp`, `analyze_shooting`).

"""
    solve_barycentric_coordinates_qp(tangent_vectors, g; compute_condition=false, return_system=false)

Shared Gram-matrix-assembly and simplex-QP-solve core of `analysis`: given the initial
tangent vectors (dense `V × V` antisymmetric matrices, one per reference) of the
geodesics from a target measure to each reference, and the target's metric tensor `g`,
assembles `A[i,j] = Σ_{x,y} tangent_vectors[i][x,y] * tangent_vectors[j][x,y] * g[x,y]`
and solves `min_{w≥0, Σw=1} w'Aw` via Convex.jl/SCS. Factored out of `analysis` so
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
    # solve the QP
    n = size(A, 1)
    x = Variable(n)
    problem = minimize(quadform(x, A))
    # Simplex constraints
    problem.constraints = vcat(problem.constraints, [x >= 0])
    problem.constraints = vcat(problem.constraints, [sum(x) == 1])

    Convex.solve!(problem, SCS.Optimizer)
    if return_system
        return (vec(x.value), A)
    end

    vec(x.value)  # optimal solution
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
