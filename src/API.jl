# Unified entry points. One public function per task with a `method` keyword selecting
# the numerical algorithm; the method-specific implementations (`geodesic_socp`,
# `barycenter_socp`, `analyze_socp`, `analyze_shooting`, `discrete_transport`, ...) stay
# defined and tested but are not exported.

const GEODESIC_METHODS   = (:socp, :shooting, :chambolle_pock)
const BARYCENTER_METHODS = (:socp, :chambolle_pock)
const ANALYSIS_METHODS   = (:socp, :shooting, :chambolle_pock)

_check_method(method, allowed, what) =
    method in allowed || throw(ArgumentError("$what: method must be one of $(allowed), got :$method"))

"""
    geodesic(G::MarkovGraph, ρA, ρB; method=:socp, kwargs...) -> GeodesicSolution

The discrete transport geodesic between densities `ρA` and `ρB` on `G`, by one of:

- `:socp` (default): a single second-order-cone program (`geodesic_socp`). Handles any
  densities, including boundary-supported ones; time-discretization error `O(1/N)`.
  Keywords: `N`, `optimizer`, `silent`.
- `:shooting`: Newton shooting on the Hamiltonian flow (`log_map`), then the flow is
  integrated to produce the path. Exact in time, fastest, but requires strictly positive
  endpoints (`PositivityFloorError`/assertion otherwise). Keywords: `nsteps`, `tol`,
  `maxiters`, `φ0_init`.
- `:chambolle_pock`: the paper's Galerkin-discretised primal-dual iteration
  (`discrete_transport`). Reference implementation; slowest. Keywords: `N`, `tol`,
  `maxiters`, `σ`, `τ`.

All methods return a `GeodesicSolution`. Its `W2` is the squared distance; `ρ` is the
`n × (steps+1)` density path and `m` the `|E| × steps` momentum path. The endpoint
potentials `φ0`, `φ1` are the gradients of `W2` with respect to each endpoint (the SOCP's
duals); for `:shooting` they are derived from the flow's velocity potentials
(`φ0 = -2φ(0)`, `φ1 = 2φ(1)`), and for `:chambolle_pock` they are not available
(`NaN`-filled). Caveat for `:chambolle_pock`: its momentum path agrees with the other
methods away from the endpoints (to `O(1/N)`), but the first-interval momentum `m0` is
a boundary-cell artifact of the Galerkin discretization, off by roughly 50% and not
converging with `N`; it is what the paper's momentum-based `analysis` uses, and is
self-consistent there, but it is not the initial momentum in the sense the other two
methods return. `status` is the JuMP termination status for `:socp` and `:converged` for
the other methods; `solvetime` is wall-clock seconds.
"""
function geodesic(G::MarkovGraph, ρA::AbstractVector, ρB::AbstractVector; method::Symbol=:socp, kwargs...)
    _check_method(method, GEODESIC_METHODS, "geodesic")
    method == :socp     && return geodesic_socp(G, ρA, ρB; kwargs...)
    method == :shooting && return _geodesic_shooting(G, ρA, ρB; kwargs...)
    return _geodesic_chambolle_pock(G, ρA, ρB; kwargs...)
end

function _geodesic_shooting(G::MarkovGraph, ρA, ρB; nsteps::Int=150, kwargs...)
    t0 = time()
    r = log_map(G, ρA, ρB; nsteps=nsteps, kwargs...)
    ρ_path, φ_path = integrate_hamiltonian(G, ρA, r.φ0; nsteps=nsteps)
    m = reduce(hcat, (metric_tensor(G, ρ_path[:, t]) .* graph_gradient(G, φ_path[:, t]) for t in 1:nsteps))
    # W2-gradient convention for the endpoint potentials, matching the SOCP's duals.
    φ0 = -2 .* r.φ0
    φ1 =  2 .* φ_path[:, end]
    return GeodesicSolution(r.W2, ρ_path, m, m[:, 1], φ0, φ1, :converged, time() - t0)
end

function _geodesic_chambolle_pock(G::MarkovGraph, ρA, ρB; kwargs...)
    t0 = time()
    a = discrete_transport(Matrix(G.Q), ρA, ρB; kwargs...)
    ρ = permutedims(a.vector.ρ)                       # (N+1) × n  ->  n × (N+1)
    N = size(a.vector.m, 1)
    m = [a.vector.m[t, x, y] for (x, y) in G.E, t in 1:N]   # |E| × N, oriented as G.E
    nan = fill(NaN, G.n)
    return GeodesicSolution(action(a), ρ, m, m[:, 1], nan, nan, :converged, time() - t0)
end

"""
    transport_cost(G::MarkovGraph, ρA, ρB; method=:socp, kwargs...) -> Float64

The discrete transport distance `𝒲(ρA, ρB)` (not squared): `sqrt(geodesic(...).W2)`.
See [`geodesic`](@ref) for the methods and keywords.
"""
transport_cost(G::MarkovGraph, ρA::AbstractVector, ρB::AbstractVector; kwargs...) =
    sqrt(geodesic(G, ρA, ρB; kwargs...).W2)

"""
    barycenter(G::MarkovGraph, refs, λ; method=:socp, kwargs...) -> (ν, J, info)

The discrete transport barycenter of the reference densities `refs` (a vector of
densities on `G`) with weights `λ`, i.e. the minimizer of `J(ν) = Σᵢ λᵢ 𝒲²(refsᵢ, ν)`, by:

- `:socp` (default): one joint second-order-cone program (`barycenter_socp`), solved to
  its global optimum. `info = (; geodesics)` holds one `GeodesicSolution` per reference
  with `λᵢ > 0`. Keywords: `N`, `optimizer`, `silent`.
- `:chambolle_pock`: the paper's intrinsic gradient descent with Chambolle-Pock
  geodesics (the `barycenter(M, weights, Q)` method). `info = (; norm_diffs, variances)`
  are the descent's per-iteration statistics. Keywords: `h`, `maxiters`, `tol`,
  `geodesic_tol`, `geodesic_steps`, `verbose`, and the rest of that method's options.
  `J` is evaluated afterwards with Chambolle-Pock geodesics at the same settings.

Returns the barycenter `ν`, the objective value `J` at `ν`, and the method-specific
`info` named tuple.
"""
function barycenter(G::MarkovGraph, refs::Vector{<:AbstractVector}, λ::AbstractVector;
                    method::Symbol=:socp, kwargs...)
    _check_method(method, BARYCENTER_METHODS, "barycenter")
    if method == :socp
        ν, J, geodesics = barycenter_socp(G, refs, λ; kwargs...)
        return ν, J, (; geodesics)
    end
    kw = Dict{Symbol,Any}(kwargs)
    geodesic_steps = get(kw, :geodesic_steps, 100)
    geodesic_tol   = get(kw, :geodesic_tol, 1e-10)
    Q = Matrix(G.Q)
    ν, norm_diffs, variances = barycenter(reduce(hcat, refs), λ, Q; return_stats=true, kwargs...)
    J = sum(λ[i] * action(discrete_transport(Q, refs[i], ν; N=geodesic_steps, tol=geodesic_tol))
            for i in eachindex(refs) if λ[i] > 0)
    return ν, J, (; norm_diffs, variances)
end

"""
    analysis(G::MarkovGraph, target, refs; method=:socp, kwargs...) -> λ̂ (or (λ̂, A))

Recover the barycentric coordinates of `target` with respect to the reference densities
`refs`: compute the geodesic from `target` to each reference, build the Gram matrix of
their tangent vectors at `target`, and solve `min_{λ∈Δ} λᵀAλ`. The geodesic solver is:

- `:socp` (default): `analyze_socp` — endpoint potentials from the geodesic SOCP.
  Keywords: `N`, `optimizer`, `convention`.
- `:shooting`: `analyze_shooting` — potentials from `log_map`; interior data only.
  Keywords: `nsteps`, `tol`, `φ0_inits`.
- `:chambolle_pock`: the paper's `analysis(ν, M, Q)` — initial momenta from
  `discrete_transport`. Keywords: `N`, `tol`.

Common keywords: `compute_condition`, `return_system` (also return the Gram matrix `A`).

A barycenter is recovered to solver tolerance only by the method (and time
resolution) that synthesized it; other methods recover it to their discretization error,
because each checks stationarity in its own discrete convention.
"""
function analysis(G::MarkovGraph, target::AbstractVector, refs::Vector{<:AbstractVector};
                  method::Symbol=:socp, kwargs...)
    _check_method(method, ANALYSIS_METHODS, "analysis")
    method == :socp     && return analyze_socp(G, target, refs; kwargs...)
    method == :shooting && return analyze_shooting(G, target, refs; kwargs...)
    return analysis(target, reduce(hcat, refs), Matrix(G.Q); kwargs...)
end
