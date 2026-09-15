"""
    weighted_laplacian(G::MarkovGraph, ν) -> SparseMatrixCSC

The `n × n` weighted graph Laplacian `L_θ(ν) = ∇ᵀ Diag(κ ∘ θ(ν)) ∇`, i.e.
`(L φ)(x) = Σ_y κ_{xy} θ(ν_x, ν_y) (φ(x) − φ(y))`. Symmetric positive semidefinite with
kernel spanned by the constants, and related to the Hamiltonian flow by
`π ∘ ρ̇ = L_θ(ν) φ` (equivalently `ρ̇ = −div(θ(ν)∘∇φ)`, see `hamiltonian_flow`).
"""
function weighted_laplacian(G::MarkovGraph, ν::AbstractVector)
    w = G.κ .* metric_tensor(G, ν)
    I = Int[]; J = Int[]; V = Float64[]
    for (e, (x, y)) in enumerate(G.E)
        push!(I, x); push!(J, x); push!(V,  w[e])
        push!(I, y); push!(J, y); push!(V,  w[e])
        push!(I, x); push!(J, y); push!(V, -w[e])
        push!(I, y); push!(J, x); push!(V, -w[e])
    end
    return sparse(I, J, V, G.n, G.n)
end

"""
    solve_weighted_laplacian(G::MarkovGraph, ν, b) -> φ

Solve `L_θ(ν) φ = b` for the unique solution in the gauge `⟨φ, 1⟩_π = 0`. `b` must be
orthogonal to the constants (`sum(b) ≈ 0`), which every right-hand side arising here
is (`π ∘ (target − ν)` for two probability densities, or `∇ᵀ(κ ∘ m)` for any momentum).
Implemented via the rank-one regularization `(L + π πᵀ) φ = b`: for such `b` this has
the same solution as `L φ = b` and enforces the gauge automatically.
"""
function solve_weighted_laplacian(G::MarkovGraph, ν::AbstractVector, b::AbstractVector)
    @assert abs(sum(b)) ≤ 1e-8 * max(1.0, maximum(abs, b)) "right-hand side must be orthogonal to constants (sum(b) = $(sum(b)))"
    A = Matrix(weighted_laplacian(G, ν)) .+ G.π * G.π'   # n is at most a few hundred; dense Cholesky is simplest
    return cholesky(Symmetric(A)) \ b
end

"""
    momentum_to_potential(G::MarkovGraph, ν, m) -> φ

Recover the potential `φ` with `m = θ(ν) ∘ ∇φ` (gauge `⟨φ, 1⟩_π = 0`) from a momentum
edge vector `m`, by solving `L_θ(ν) φ = ∇ᵀ(κ ∘ m)`. If `m` is not exactly a gradient
field this returns the `θ(ν)`-weighted least-squares projection, i.e. the potential of
the gradient part of `m` in the Hodge sense.
"""
function momentum_to_potential(G::MarkovGraph, ν::AbstractVector, m::AbstractVector)
    @assert length(m) == length(G.E)
    b = zeros(G.n)
    for (e, (x, y)) in enumerate(G.E)
        b[x] += G.κ[e] * m[e]
        b[y] -= G.κ[e] * m[e]
    end
    return solve_weighted_laplacian(G, ν, b)
end

"""
    exp_map(G::MarkovGraph, ν, tangent; t=1.0, nsteps=150, kind=:auto, floor_rtol=1e-6) -> ρ_end

Module 3.2: the Riemannian exponential map at `ν`. Integrates the Hamiltonian flow
(`integrate_hamiltonian`) from `(ν, φ0)` for time `t` and returns the endpoint density.

`tangent` is either a potential `φ0 ∈ Rⁿ` (`kind=:potential`) or a momentum
`m0 ∈ R^{|E|}` (`kind=:momentum`), in which case `φ0` is recovered first with
`momentum_to_potential`. With `kind=:auto` (default) the kind is inferred from the
length, which is ambiguous when `n == |E|` (e.g. a cycle); then `kind` must be given.
The potential is gauge-fixed to `⟨φ0, 1⟩_π = 0` before integrating (this does not
change the flow's `ρ`).

Requires `ν` strictly positive (see `ρ_floor`); errors, rather than returning garbage,
if the flow hits the positivity floor before time `t`.
"""
function exp_map(G::MarkovGraph, ν::AbstractVector, tangent::AbstractVector;
                 t::Float64=1.0, nsteps::Int=150, kind::Symbol=:auto, floor_rtol::Float64=1e-6)
    n, nE = G.n, length(G.E)
    if kind == :auto
        if length(tangent) == n && n != nE
            kind = :potential
        elseif length(tangent) == nE && n != nE
            kind = :momentum
        elseif n == nE
            throw(ArgumentError("n == |E| == $n, so the kind of `tangent` cannot be inferred; pass kind=:potential or kind=:momentum"))
        else
            throw(ArgumentError("tangent has length $(length(tangent)); expected n=$n (potential) or |E|=$nE (momentum)"))
        end
    end
    φ0 = if kind == :potential
        @assert length(tangent) == n
        tangent .- dot(tangent, G.π)   # gauge ⟨φ0, 1⟩_π = 0 (π sums to one)
    elseif kind == :momentum
        momentum_to_potential(G, ν, tangent)
    else
        throw(ArgumentError("kind must be :auto, :potential or :momentum, got $kind"))
    end
    ρ_path, _ = integrate_hamiltonian(G, ν, φ0; nsteps=nsteps, T=t, floor_rtol=floor_rtol)
    return ρ_path[:, end]
end
