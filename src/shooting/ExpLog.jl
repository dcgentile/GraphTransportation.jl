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

# Reduced coordinates for the shooting unknown: z ∈ R^{n-1} ↦ φ0 ∈ Rⁿ with the gauge
# ⟨φ0, 1⟩_π = 0 solved for the last component. Removes the flow's gauge freedom, so
# the Newton system below is square and (generically) nonsingular.
function _reduced_to_potential(G::MarkovGraph, z::AbstractVector)
    n = G.n
    last = -dot(view(G.π, 1:n-1), z) / G.π[n]
    return vcat(z, last)
end

"""
    log_map(G::MarkovGraph, ν, target; φ0_init=nothing, tol=1e-9, maxiters=50, nsteps=150,
            floor_rtol=1e-6, verbose=false) -> (; φ0, m0, W2, iters, residual)

Module 3.3: the Riemannian logarithm at `ν`, by single shooting. Solves
`F(φ0) := ρ(1; ν, φ0) − target = 0` with a damped Newton iteration over the mean-zero
potentials (`n − 1` unknowns, the gauge `⟨φ0, 1⟩_π = 0` and the mass constraint each
removing one dimension), with the Jacobian computed by `ForwardDiff` through
`integrate_hamiltonian` and a backtracking line search on `‖F‖_π`.

Initialization is the linearized geodesic `L_θ(ν) φ0 = π ∘ (target − ν)`, exact to first
order in `target − ν` (spec §3.3), unless `φ0_init` (a potential) is given, e.g. from a
previous solve at a nearby base point (spec §3.4 warm-start).

Returns the potential `φ0`, the momentum `m0 = θ(ν) ∘ ∇φ0`, the squared distance
`W2 = 2H(ν, φ0)`, the Newton iteration count, and the final residual `‖F‖_π`. Errors if
the iteration has not reached `tol` after `maxiters` steps, or if the shooting
trajectory persistently hits the positivity floor (fall back to `geodesic_socp`, or
mollify, spec §3.5). Requires `ν` and `target` strictly positive (see `ρ_floor`).
"""
function log_map(G::MarkovGraph, ν::AbstractVector, target::AbstractVector;
                 φ0_init=nothing, tol::Float64=1e-9, maxiters::Int=50, nsteps::Int=150,
                 floor_rtol::Float64=1e-6, verbose::Bool=false)
    n = G.n
    floor_val = ρ_floor(G; rtol=floor_rtol)
    @assert minimum(ν) > floor_val && minimum(target) > floor_val "log_map requires strictly positive endpoints (see ρ_floor)"
    @assert abs(dot(ν, G.π) - 1) < 1e-8 && abs(dot(target, G.π) - 1) < 1e-8 "endpoints must be probability densities"

    sqrtπ = sqrt.(G.π)
    # Full residual (all n components; the last is redundant but harmless for the norm).
    function shoot(z)
        φ0 = _reduced_to_potential(G, z)
        ρ_path, _ = integrate_hamiltonian(G, ν, φ0; nsteps=nsteps, T=1.0, floor_rtol=floor_rtol)
        return ρ_path[:, end] .- target
    end
    F_reduced(z) = shoot(z)[1:n-1]
    resnorm(F) = norm(F .* sqrtπ)

    φ0 = if φ0_init === nothing
        solve_weighted_laplacian(G, ν, G.π .* (target .- ν))
    else
        φ0_init .- dot(φ0_init, G.π)
    end
    z = φ0[1:n-1]

    # The linearized guess can overshoot through the positivity floor for far-apart
    # endpoints (it is only first-order accurate); damp it until the first shot survives.
    local F
    for k in 0:12
        F = try
            shoot(z)
        catch err
            err isa ErrorException || rethrow()
            k == 12 && error("log_map: no admissible initial potential found (endpoints too far apart for shooting); " *
                             "fall back to geodesic_socp or mollify the endpoints (spec §3.5).")
            z ./= 2
            nothing
        end
        F === nothing || break
    end
    r = resnorm(F)
    iters = 0
    while r > tol
        iters ≥ maxiters && error("log_map: Newton did not converge in $maxiters iterations (residual $r > tol $tol); " *
                                  "fall back to geodesic_socp or mollify the endpoints (spec §3.5).")
        J = ForwardDiff.jacobian(F_reduced, z)
        δ = -(J \ F[1:n-1])

        # Backtracking on ‖F‖_π; a step whose trajectory hits the positivity floor counts
        # as a failed step and is shortened the same way.
        α = 1.0
        accepted = false
        for _ in 1:12
            z_try = z .+ α .* δ
            F_try = try
                shoot(z_try)
            catch err
                err isa ErrorException || rethrow()
                nothing
            end
            if F_try !== nothing && resnorm(F_try) ≤ (1 - 1e-4 * α) * r
                z, F, r = z_try, F_try, resnorm(F_try)
                accepted = true
                break
            end
            α /= 2
        end
        accepted || error("log_map: line search failed at iteration $(iters + 1) (residual $r); " *
                          "the target may be too far from ν for single shooting, or the geodesic leaves the positive cone.")
        iters += 1
        verbose && @info "log_map" iter=iters residual=r step=α
    end

    φ0 = _reduced_to_potential(G, z)
    m0 = metric_tensor(G, ν) .* graph_gradient(G, φ0)
    W2 = 2 * hamiltonian(G, ν, φ0)
    return (; φ0, m0, W2, iters, residual=r)
end

"""
    analyze_shooting(G::MarkovGraph, target, refs; nsteps=150, tol=1e-9, φ0_inits=nothing,
                     compute_condition=false, return_system=false) -> λ̂ (or (λ̂, A))

Module 4's `:shooting` backend: like `analyze_socp`, but each reference's potential
comes from `log_map(G, target, ref)` (the Hamiltonian velocity potential `φ0` at
`target`) instead of the geodesic SOCP's endpoint dual. The Gram matrix and simplex QP
are shared (`potential_gram_qp`). Requires strictly positive `target` and `refs`.

`φ0_inits`, if given, is a vector of warm-start potentials, one per reference (spec §3.4).

Note that this checks stationarity in the Hamiltonian flow's (RK4, `nsteps`)
discretization, which differs from both `barycenter_socp`'s and the Chambolle-Pock
descent's. A barycenter synthesized by either of those is therefore recovered only to
`O(h)` in *that* method's time step, not to solver tolerance — see
`SOCP_ANALYSIS_SPEC.md` §8 for why synthesis and analysis conventions must match.
"""
function analyze_shooting(G::MarkovGraph, target::AbstractVector, refs::Vector{<:AbstractVector};
                          nsteps::Int=150, tol::Float64=1e-9, φ0_inits=nothing,
                          compute_condition::Bool=false, return_system::Bool=false)
    potentials = map(eachindex(refs)) do i
        init = φ0_inits === nothing ? nothing : φ0_inits[i]
        log_map(G, target, refs[i]; nsteps=nsteps, tol=tol, φ0_init=init).φ0
    end
    return potential_gram_qp(G, target, potentials;
                             compute_condition=compute_condition, return_system=return_system)
end

"""
    log_map_mollified(G::MarkovGraph, ν, target; εs=(1e-2, 1e-3, 1e-4), kwargs...)
        -> (; W2, W, φ0, m0, fit, εs, Ws, approximate=true)

Module 3.5: `log_map` for endpoints with zero or near-zero entries, where shooting
cannot be run directly. Both endpoints are mollified toward the uniform density,
`ρ_ε = (1−ε)ρ + ε·𝟙` (still a probability density), `log_map` is run for each `ε` in
`εs` (warm-starting each from the previous, coarser `ε`), and the distance is
extrapolated to `ε → 0` by a least-squares fit `W(ε) ≈ W₀ + a·√ε` (the mollification
perturbs the endpoints by `O(√ε)` in `𝒲`). Returns the extrapolated `W` and `W2 = W²`,
plus the potential/momentum from the smallest solved `ε` (at the mollified base point
`ν_ε`, so only approximately a tangent at `ν`), the fit coefficients `(W₀, a)`, and the
raw per-`ε` distances. Levels whose shooting fails (very small `ε` makes the
near-boundary geodesic stiff) are skipped with a warning as long as two remain.
Results are flagged `approximate=true`; for boundary-supported data `geodesic_socp`
needs no such fallback and is the reference. Empirically the raw `W` at the smallest
`ε` is often already as accurate as the extrapolation (the mollification error can
decay faster than `√ε`), so both are returned. Remaining `kwargs` go to `log_map`.
"""
function log_map_mollified(G::MarkovGraph, ν::AbstractVector, target::AbstractVector;
                           εs=(1e-2, 1e-3, 1e-4), kwargs...)
    @assert length(εs) ≥ 2 "need at least two ε values to extrapolate"
    εs = sort(collect(Float64, εs); rev=true)
    mollify(ρ, ε) = (1 - ε) .* ρ .+ ε
    Ws = Float64[]
    used = Float64[]
    r = nothing
    for ε in εs
        # Very small ε makes the near-boundary geodesic stiff for single shooting; skip
        # such levels rather than fail the whole extrapolation, as long as two remain.
        r_ε = try
            log_map(G, mollify(ν, ε), mollify(target, ε); φ0_init=(r === nothing ? nothing : r.φ0), kwargs...)
        catch err
            err isa ErrorException || rethrow()
            @warn "log_map_mollified: shooting failed at ε=$ε, skipping this level" exception=err.msg
            continue
        end
        r = r_ε
        push!(Ws, sqrt(r.W2)); push!(used, ε)
    end
    length(used) ≥ 2 || error("log_map_mollified: fewer than two ε levels solved; fall back to geodesic_socp")
    X = hcat(ones(length(used)), sqrt.(used))
    W0, a = X \ Ws
    return (; W2=W0^2, W=W0, φ0=r.φ0, m0=r.m0, fit=(W0, a), εs=used, Ws, approximate=true)
end
