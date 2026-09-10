"""
Module 3 primitives (`spec.txt`): the Hamiltonian ODE system underlying the
exponential/logarithmic maps. Valid only for **strictly positive** densities
(`ρ_floor`-guarded); see `spec.txt` §3 for the boundary-case mollification fallback
(not yet implemented) and Module 1 (`geodesic_socp`) for the general case.

State is `(ρ, φ) ∈ Rⁿ × Rⁿ`. Uses the geometric mean `θ(s,t) = √(st)` throughout,
matching Modules 0-2; this is not configurable.

The equations of motion below are Hamilton's equations for the `⟨·,·⟩_π`-weighted
pairing (`ρ̇(z) = (1/π(z)) ∂H/∂φ(z)`, `φ̇(z) = -(1/π(z)) ∂H/∂ρ(z)`), re-derived here
by direct differentiation of `H` rather than taken on faith from `spec.txt` — they
match spec.txt's stated formulas exactly, including the `ρ̇ = -div(θ(ρ)∘∇φ)` identity,
which is what lets `ρ̇` reuse `graph_divergence`/`graph_gradient` directly.
"""

"""
    ρ_floor(G::MarkovGraph; rtol=1e-6) -> Float64

Minimum density Module 3's ODE machinery will tolerate: `rtol` relative to `min(π)`,
per `spec.txt`'s guard (`@assert minimum(ρ) > ρ_floor`). Below this, `θ_geo` and its
partial derivative `∂₁θ_geo(s,t) = ½√(t/s)` are numerically unsafe (division by a
near-zero `s`), and the caller should fall back to `geodesic_socp` or a mollified
approximation (`spec.txt` §3.5, not yet implemented) instead.
"""
ρ_floor(G::MarkovGraph; rtol::Float64=1e-6) = rtol * minimum(G.π)

"""
    hamiltonian(G::MarkovGraph, ρ, φ) -> Float64

`H(ρ,φ) = ½ Σ_e θ(ρ_x,ρ_y) (∇φ)_e² κ_e`. Requires `ρ` strictly positive (see `ρ_floor`).
"""
function hamiltonian(G::MarkovGraph, ρ::AbstractVector, φ::AbstractVector)
    ∇φ = graph_gradient(G, φ)
    θ = [sqrt(ρ[x] * ρ[y]) for (x, y) in G.E]
    return 0.5 * sum(G.κ .* θ .* ∇φ .^ 2)
end

"""
    hamiltonian_flow(G::MarkovGraph, ρ, φ) -> (ρ̇, φ̇)

Equations of motion (Module 3.1):

    ρ̇(x) = Σ_y θ(ρ(x),ρ(y)) (φ(x)-φ(y)) Q(x,y)  =  -div(θ(ρ)∘∇φ)(x)
    φ̇(x) = -½ Σ_y ∂₁θ(ρ(x),ρ(y)) (φ(x)-φ(y))² Q(x,y),   ∂₁θ_geo(s,t) = ½√(t/s)

Requires `ρ` strictly positive (see `ρ_floor`); `∂₁θ_geo` divides by `ρ(x)`.
"""
function hamiltonian_flow(G::MarkovGraph, ρ::AbstractVector, φ::AbstractVector)
    ∇φ = graph_gradient(G, φ)
    θ = [sqrt(ρ[x] * ρ[y]) for (x, y) in G.E]
    ρ̇ = .-graph_divergence(G, θ .* ∇φ)

    φ̇ = zeros(G.n)
    for (e, (x, y)) in enumerate(G.E)
        Δφ² = ∇φ[e]^2
        φ̇[x] -= 0.5 * (0.5 * sqrt(ρ[y] / ρ[x])) * Δφ² * G.Q[x, y]
        φ̇[y] -= 0.5 * (0.5 * sqrt(ρ[x] / ρ[y])) * Δφ² * G.Q[y, x]
    end
    return ρ̇, φ̇
end

"""
    integrate_hamiltonian(G::MarkovGraph, ρ0, φ0; nsteps=150, T=1.0, floor_rtol=1e-6)
        -> (ρ_path, φ_path)

Integrate the Module 3 Hamiltonian flow forward from `(ρ0, φ0)` over `[0, T]` using
fixed-step classical RK4 (`spec.txt` explicitly sanctions RK4 over a symplectic
integrator: "symplecticity is a nicety, not a requirement"). Returns the full paths
as `n × (nsteps+1)` matrices.

Positivity guard: if any density in the proposed next step would fall below
`ρ_floor(G; rtol=floor_rtol)`, the step is halved (up to 4 times) before giving up
and erroring — per `spec.txt`'s guidance to fall back to `geodesic_socp` on
persistent violation, which callers should catch and act on.
"""
function integrate_hamiltonian(G::MarkovGraph, ρ0::AbstractVector, φ0::AbstractVector;
                                nsteps::Int=150, T::Float64=1.0, floor_rtol::Float64=1e-6)
    floor_val = ρ_floor(G; rtol=floor_rtol)
    @assert minimum(ρ0) > floor_val "ρ0 violates the positivity floor (Module 3 requires strictly positive densities)"

    n = G.n
    ρ_path = zeros(n, nsteps + 1)
    φ_path = zeros(n, nsteps + 1)
    ρ_path[:, 1] = ρ0
    φ_path[:, 1] = φ0

    h = T / nsteps
    ρ, φ = copy(ρ0), copy(φ0)
    for i in 1:nsteps
        ρ, φ = _advance_interval(G, ρ, φ, h, floor_val, 4)
        ρ_path[:, i+1] = ρ
        φ_path[:, i+1] = φ
    end
    return ρ_path, φ_path
end

function _rk4_step(G::MarkovGraph, ρ, φ, h)
    k1ρ, k1φ = hamiltonian_flow(G, ρ, φ)
    k2ρ, k2φ = hamiltonian_flow(G, ρ .+ (h/2) .* k1ρ, φ .+ (h/2) .* k1φ)
    k3ρ, k3φ = hamiltonian_flow(G, ρ .+ (h/2) .* k2ρ, φ .+ (h/2) .* k2φ)
    k4ρ, k4φ = hamiltonian_flow(G, ρ .+ h .* k3ρ, φ .+ h .* k3φ)
    ρ_next = ρ .+ (h/6) .* (k1ρ .+ 2 .* k2ρ .+ 2 .* k3ρ .+ k4ρ)
    φ_next = φ .+ (h/6) .* (k1φ .+ 2 .* k2φ .+ 2 .* k3φ .+ k4φ)
    return ρ_next, φ_next
end

# Advance by exactly Δt, bisecting (and recursing on each half) whenever a step would
# cross the positivity floor, so a triggered guard shrinks the *local* step size
# without changing the *total* elapsed time - unlike naively retrying with a smaller h
# and accepting a shorter advance, which would silently desync the integrator's clock
# from the nsteps*h = T the caller expects.
function _advance_interval(G::MarkovGraph, ρ, φ, Δt, floor_val, depth)
    # An RK4 stage evaluates hamiltonian_flow at *intermediate* proposed states
    # (ρ + (Δt/2)k1, etc.), which can dip below the floor - and hence hit sqrt of a
    # negative θ argument - even when the accepted output of the step would not have.
    # That surfaces as a DomainError from inside hamiltonian_flow, not as an
    # out-of-range ρ_next we could check post-hoc, so it needs to trigger the same
    # bisection as an explicit floor violation.
    ρ_next, φ_next = try
        _rk4_step(G, ρ, φ, Δt)
    catch err
        err isa DomainError || rethrow()
        (fill(-Inf, length(ρ)), φ)
    end
    minimum(ρ_next) > floor_val && return ρ_next, φ_next
    depth <= 0 && error("Hamiltonian integration hit the positivity floor after repeated step " *
                         "halving; fall back to geodesic_socp for this instance.")
    ρ_mid, φ_mid = _advance_interval(G, ρ, φ, Δt / 2, floor_val, depth - 1)
    return _advance_interval(G, ρ_mid, φ_mid, Δt / 2, floor_val, depth - 1)
end
