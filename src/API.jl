# Unified entry points. One public function per task with a `method` keyword selecting
# the numerical algorithm; the method-specific implementations (`geodesic_socp`,
# `barycenter_socp`, `analyze_socp`, `analyze_shooting`, `discrete_transport`, ...) stay
# defined and tested but are not exported.

const GEODESIC_METHODS   = (:socp, :shooting, :chambolle_pock, :sinkhorn)
const BARYCENTER_METHODS = (:socp, :chambolle_pock, :sinkhorn)
const ANALYSIS_METHODS   = (:socp, :shooting, :chambolle_pock, :sinkhorn)

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
- `:sinkhorn`: the **entropic displacement interpolation for a ground cost**: the path
  is the entropic barycenter of the two endpoints at weights `(1−t, t)` for
  `t = 0, 1/N, …, 1` (the two-reference special case of `barycenter(...; method=:sinkhorn)`),
  a different object from the discrete transport geodesic. Requires `cost` (see
  `ground_cost`) and `epsilon`; keywords `N` (default 10) and `iters` (Sinkhorn budget,
  default 256). `W2` is the entropic transport cost `⟨cost, P⟩` of the plan between the
  endpoints (no entropy term). The path's end columns are the *blurred* endpoints the
  Sinkhorn barycenter returns, not `ρA`/`ρB` exactly; `m`, `φ0`, `φ1` are `NaN`-filled.

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
    method == :sinkhorn && return _geodesic_sinkhorn(G, ρA, ρB; kwargs...)
    return _geodesic_chambolle_pock(G, ρA, ρB; kwargs...)
end

function _geodesic_sinkhorn(G::MarkovGraph, ρA, ρB; N::Int=10, cost=nothing, epsilon=nothing, iters::Int=256)
    t0 = time()
    cost    === nothing && throw(ArgumentError("geodesic(method=:sinkhorn) requires cost= (see ground_cost)"))
    epsilon === nothing && throw(ArgumentError("geodesic(method=:sinkhorn) requires epsilon="))
    ρ = zeros(G.n, N + 1)
    for (k, t) in enumerate(range(0.0, 1.0, length=N + 1))
        ρ[:, k] = _barycenter_sinkhorn(G, [ρA, ρB], [1 - t, t]; cost=cost, epsilon=epsilon, iters=iters)[1]
    end
    K = regularize_cost(cost, epsilon)
    P = _sinkhorn_plan(K, ρA .* G.π, ρB .* G.π; iters=iters)
    W2 = dot(cost, P)
    nanE = fill(NaN, length(G.E), N); nan = fill(NaN, G.n)
    return GeodesicSolution(W2, ρ, nanE, nanE[:, 1], nan, nan, :converged, time() - t0)
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
- `:sinkhorn`: the **entropically regularized Wasserstein barycenter for a ground cost**
  (Benamou et al. 2015; Bonneel, Peyré & Cuturi 2016; `sinkhorn_barycenter`). This is a
  different object from the discrete transport barycenter of the other methods: it depends
  on the choice of `cost` (see `ground_cost`) and on `epsilon`, and even for a single
  reference it returns a blurred copy of that reference. Both `cost` and `epsilon` are
  required keywords; `iters` (default 256) is the Sinkhorn budget. `J = Σᵢ λᵢ ⟨cost, Pᵢ⟩`
  over the entropic plans `Pᵢ` (no entropy term) and is not comparable with the `J` of the
  other methods; `info = (; cost, epsilon, iters, marginal_errors)`. `refs` are densities
  with respect to `π` as elsewhere (converted to probability vectors internally).

Returns the barycenter `ν`, the objective value `J` at `ν`, and the method-specific
`info` named tuple.
"""
function barycenter(G::MarkovGraph, refs::Vector{<:AbstractVector}, λ::AbstractVector;
                    method::Symbol=:socp, kwargs...)
    _check_method(method, BARYCENTER_METHODS, "barycenter")
    if method == :socp
        ν, J, geodesics = barycenter_socp(G, refs, λ; kwargs...)
        return ν, J, (; geodesics)
    elseif method == :sinkhorn
        return _barycenter_sinkhorn(G, refs, λ; kwargs...)
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
- `:sinkhorn`: **Wasserstein barycentric coordinates for a ground cost** (Bonneel, Peyré
  & Cuturi 2016; `simplex_regression`): L-BFGS over the simplex on
  `½‖P(λ) − target‖²`, with `P(λ)` the entropic barycenter of `refs`, differentiated
  through the Sinkhorn iterations. Not a Gram-matrix method, so `return_system` and
  `compute_condition` are not supported (an error, not silently ignored). Requires
  `cost` and `epsilon`; keywords `iters` (Sinkhorn budget, default 256) and `α0`
  (initial pre-softmax point, default `0`, i.e. uniform weights). The result depends on
  `cost` and `epsilon`, and recovers exactly the weights of a barycenter synthesized by
  `barycenter(...; method=:sinkhorn)` with the same `cost`, `epsilon` and `iters`.

Common keywords (Gram-matrix methods): `compute_condition`, `return_system` (also return
the Gram matrix `A`).

A barycenter is recovered to solver tolerance only by the method (and time
resolution) that synthesized it; other methods recover it to their discretization error,
because each checks stationarity in its own discrete convention.
"""
function analysis(G::MarkovGraph, target::AbstractVector, refs::Vector{<:AbstractVector};
                  method::Symbol=:socp, kwargs...)
    _check_method(method, ANALYSIS_METHODS, "analysis")
    method == :socp     && return analyze_socp(G, target, refs; kwargs...)
    method == :shooting && return analyze_shooting(G, target, refs; kwargs...)
    method == :sinkhorn && return _analysis_sinkhorn(G, target, refs; kwargs...)
    return analysis(target, reduce(hcat, refs), Matrix(G.Q); kwargs...)
end

function _analysis_sinkhorn(G::MarkovGraph, target, refs; cost=nothing, epsilon=nothing, iters::Int=256,
                            α0=zeros(length(refs)), compute_condition::Bool=false, return_system::Bool=false)
    cost    === nothing && throw(ArgumentError("analysis(method=:sinkhorn) requires cost= (see ground_cost)"))
    epsilon === nothing && throw(ArgumentError("analysis(method=:sinkhorn) requires epsilon="))
    (compute_condition || return_system) &&
        throw(ArgumentError("analysis(method=:sinkhorn) is not a Gram-matrix method; compute_condition/return_system are not available"))
    μ = reduce(hcat, (r .* G.π for r in refs))
    return simplex_regression(μ, target .* G.π, cost, epsilon; iters=iters, α0=α0)
end

function _barycenter_sinkhorn(G::MarkovGraph, refs, λ; cost=nothing, epsilon=nothing, iters::Int=256)
    cost    === nothing && throw(ArgumentError("barycenter(method=:sinkhorn) requires cost= (see ground_cost)"))
    epsilon === nothing && throw(ArgumentError("barycenter(method=:sinkhorn) requires epsilon="))
    size(cost) == (G.n, G.n) || throw(ArgumentError("cost must be $(G.n)×$(G.n)"))
    μ = reduce(hcat, (r .* G.π for r in refs))              # densities -> probability vectors
    p = sinkhorn_barycenter(λ, μ, nothing, cost, epsilon; iters=iters)
    K = regularize_cost(cost, epsilon)
    active = findall(>(0), λ)
    plans = [_sinkhorn_plan(K, μ[:, i], p; iters=iters) for i in active]
    J = sum(λ[i] * dot(cost, P) for (i, P) in zip(active, plans))
    marginal_errors = [norm(vec(sum(P, dims=2)) .- μ[:, i], 1) for (i, P) in zip(active, plans)]
    return p ./ G.π, J, (; cost, epsilon, iters, marginal_errors)
end
