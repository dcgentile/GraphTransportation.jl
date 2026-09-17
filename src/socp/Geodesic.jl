"""
    GeodesicSolution

Result of solving the geodesic SOCP (`geodesic_socp`). Note the naming
convention: `W2` is the **squared** discrete transport distance; the metric
itself is `sqrt(W2)`.

Fields:
- `W2::Float64`: squared discrete transport distance, i.e. the SOCP objective value.
- `ρ::Matrix{Float64}`: the `n × (N+1)` density path, `ρ[:, i+1]` at time `i/N`.
- `m::Matrix{Float64}`: the `|E| × N` momentum path.
- `m0::Vector{Float64}`: `m[:, 1]`, the initial momentum (convenience).
- `φ0::Vector{Float64}`, `φ1::Vector{Float64}`: the discrete potentials at the two
  endpoints, read off the duals of the endpoint constraints `ρ[:,1] == ρA` and
  `ρ[:,N+1] == ρB` and divided by `π`, so that they are exactly the gradient of `W2`
  with respect to each endpoint density under the `π`-weighted pairing:
  `δW2 = ⟨φ0, δρA⟩_π + ⟨φ1, δρB⟩_π`. This is the discrete Hamilton–Jacobi
  potential (`m ≈ -½ θ ∇φ` in the continuum limit; see `_geodesic_block!`), and is
  the quantity the potential-based analysis Gram matrix (`analyze_socp`) is built from.
  Each is defined only up to an additive constant; JuMP/Clarabel return the
  representative with `⟨φ, 1⟩` fixed by the solver, which is harmless since only
  `∇φ` is ever used.
- `status`: JuMP termination status of the solve.
- `solvetime::Float64`: wall-clock solve time in seconds.
"""
struct GeodesicSolution
    W2::Float64
    ρ::Matrix{Float64}
    m::Matrix{Float64}
    m0::Vector{Float64}
    φ0::Vector{Float64}
    φ1::Vector{Float64}
    status::Any
    solvetime::Float64
end

"""
    _mean_cone!(model, θ::AdmissibleMean, ρ̄x, ρ̄y, ϑ)

Add the constraint `ϑ ≤ θ(ρ̄x, ρ̄y)` (the hypograph of the mean) in conic form:
- `GeometricMean`: `[ρ̄x, ρ̄y, √2 ϑ] ∈ RSOC`, i.e. `ϑ² ≤ ρ̄x ρ̄y`.
- `ArithmeticMean`: a linear row.
- `HarmonicMean`: `ϑ ≤ 2st/(s+t)`. Multiply by `s+t > 0`: `(s+t)ϑ ≤ 2st`. Use the identity
  `4st = (s+t)² − (s−t)²`, i.e. `2st = ((s+t)² − (s−t)²)/2`, so the inequality becomes
  `2(s+t)ϑ ≤ (s+t)² − (s−t)²`, i.e. `(s−t)² ≤ (s+t)(s+t−2ϑ)` with `s+t−2ϑ ≥ 0`. That is a
  rotated second-order cone `x·y ≥ z²` with `x = s+t`, `y = s+t−2ϑ`, `z = s−t`. JuMP's
  `RotatedSecondOrderCone` is `2·x·y ≥ ‖z‖²`, so `z` is scaled by `√2` to cancel the 2:
  `[s+t, s+t−2ϑ, √2 (s−t)]`. (The same `√2` scaling is why the geometric cone carries
  `√2 ϑ`.) Verified against `2st/(s+t)` in the tests.
- `QuadLogMean`: `K` power cones `[ρ̄x, ρ̄y, ϑ_k] ∈ PowerCone(α_k)` (`ρ̄x^{α_k} ρ̄y^{1−α_k} ≥ ϑ_k`)
  and the linear row `ϑ ≤ Σ_k w_k ϑ_k`.
`LogarithmicMean` has no finite conic representation; pass a `QuadLogMean` instead.
Every form was gated against the generalized two-node closed form
`W = (1/√2)∫ θ(1−r,1+r)^{−1/2} dr` in the tests.
"""
function _mean_cone!(model, ::GeometricMean, ρ̄x, ρ̄y, ϑ)
    @constraint(model, [ρ̄x, ρ̄y, sqrt(2) * ϑ] in RotatedSecondOrderCone())
end
function _mean_cone!(model, ::ArithmeticMean, ρ̄x, ρ̄y, ϑ)
    @constraint(model, ϑ <= (ρ̄x + ρ̄y) / 2)
end
function _mean_cone!(model, ::HarmonicMean, ρ̄x, ρ̄y, ϑ)
    @constraint(model, [ρ̄x + ρ̄y, ρ̄x + ρ̄y - 2 * ϑ, sqrt(2) * (ρ̄x - ρ̄y)] in RotatedSecondOrderCone())
end
function _mean_cone!(model, θ::QuadLogMean, ρ̄x, ρ̄y, ϑ)
    K = length(θ.α)
    ϑk = @variable(model, [1:K], lower_bound = 0)
    for k in 1:K
        @constraint(model, [ρ̄x, ρ̄y, ϑk[k]] in MOI.PowerCone(θ.α[k]))
    end
    @constraint(model, ϑ <= sum(θ.w[k] * ϑk[k] for k in 1:K))
end
_mean_cone!(model, ::LogarithmicMean, ρ̄x, ρ̄y, ϑ) =
    throw(ArgumentError("LogarithmicMean has no conic representation; use QuadLogMean(K) in the SOCP (K=8 is accurate to 1e-10)"))

"""
    _geodesic_block!(model, G, N, h, left, right; base_name="") -> (; ρ, m, ϑ, w, c_left, c_right, c_cont)

Add one geodesic's worth of variables and constraints to `model`:
density/momentum/mean/action variables, the discrete continuity equation, and the
mean-cone and action-epigraph RSOC constraints. `left` and `right` fix the two
boundary densities and may each be either a plain vector (as in `geodesic_socp`) or
themselves JuMP variables/expressions (as in `barycenter_socp`, where `right` is the
shared barycenter variable `ν`) — `@constraint(model, ρ[:, k] .== x)` accepts either.
Does not set an objective; callers combine one or more blocks' `w` fields into theirs.
The mobility `θ` is `G.mean` (see `_mean_cone!`).

Also returns the constraint references whose duals carry the potentials:
- `c_left`, `c_right`: the `n` endpoint constraints `ρ[:,1] .== left` and
  `ρ[:,N+1] .== right`. After `optimize!`, `dual.(c_left) ./ G.π` is the discrete
  potential at the left endpoint — the exact gradient of this block's contribution
  to the objective with respect to `left`, in the `π`-weighted pairing (calibrated
  against finite differences and the two-node closed form; JuMP's sign convention
  needs no flip, and if the block's action is weighted by `λ` in the objective the
  dual scales by `λ` too). Likewise `c_right`.
- `c_cont`: the `N` vectors of continuity-equation constraints. Their duals `ψ_t`
  satisfy `m[:,t] = -(1/2h) θ(ρ̄_t) ∘ ∇(ψ_t ./ π)` exactly (with `ρ̄_t` the interval
  midpoint density, not the endpoint density — using `θ(ρ0)` here is what makes the
  ratio look non-constant). Not needed by any caller today; exposed for testing.
"""
function _geodesic_block!(model, G::MarkovGraph, N::Int, h::Float64,
                           left::AbstractVector, right::AbstractVector; base_name::String="")
    n = G.n
    nE = length(G.E)

    ρ = @variable(model, [1:n, 1:(N+1)], lower_bound = 0, base_name = "ρ" * base_name)
    m = @variable(model, [1:nE, 1:N], base_name = "m" * base_name)
    ϑ = @variable(model, [1:nE, 1:N], lower_bound = 0, base_name = "ϑ" * base_name)
    w = @variable(model, [1:nE, 1:N], lower_bound = 0, base_name = "w" * base_name)

    c_left  = @constraint(model, ρ[:, 1] .== left)
    c_right = @constraint(model, ρ[:, N+1] .== right)

    # discrete continuity equation: (ρ_{t+1} - ρ_t)/h + div(m_t) = 0
    c_cont = map(1:N) do t
        divm = graph_divergence(G, m[:, t])
        @constraint(model, (ρ[:, t+1] .- ρ[:, t]) ./ h .+ divm .== 0)
    end

    # mean cone: ϑ_{e,t} ≤ θ(ρ̄_{x,t}, ρ̄_{y,t})   (form depends on the mean, see _mean_cone!)
    # action epigraph (rotated SOC): m_{e,t}² ≤ ϑ_{e,t} w_{e,t}
    for t in 1:N, (e, (x, y)) in enumerate(G.E)
        ρ̄x = (ρ[x, t] + ρ[x, t+1]) / 2
        ρ̄y = (ρ[y, t] + ρ[y, t+1]) / 2
        _mean_cone!(model, G.mean, ρ̄x, ρ̄y, ϑ[e, t])
        @constraint(model, [ϑ[e, t], w[e, t], sqrt(2) * m[e, t]] in RotatedSecondOrderCone())
    end

    return (ρ=ρ, m=m, ϑ=ϑ, w=w, c_left=c_left, c_right=c_right, c_cont=c_cont)
end

"""
    _endpoint_potentials(G, blk; weight=1.0) -> (φ0, φ1)

Read the endpoint potentials off a solved `_geodesic_block!` (see its docstring):
`dual.(c) ./ G.π ./ weight`, where `weight` is the factor multiplying this block's
action in the model objective (`λ[i]` in `barycenter_socp`, `1` in `geodesic_socp`),
so the result is always the potential of the *unweighted* geodesic.
"""
function _endpoint_potentials(G::MarkovGraph, blk; weight::Float64=1.0)
    φ0 = dual.(blk.c_left)  ./ G.π ./ weight
    φ1 = dual.(blk.c_right) ./ G.π ./ weight
    return φ0, φ1
end

"""
    _check_solved(model, what)

Throw unless the conic solve reached `OPTIMAL` or `ALMOST_OPTIMAL`. Clarabel can stop
with `ITERATION_LIMIT` or `SLOW_PROGRESS` on large or ill-conditioned instances (seen with
the quadrature-log mean's power cones at N=10 on a 49-node graph) and the iterate it
leaves behind is not a solution: it need not even have unit mass. Returning it silently
produced a barycenter with total mass 0.05. Callers who want the partial iterate can pass
`check=false`.
"""
function _check_solved(model, what)
    st = termination_status(model)
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) && return st
    throw(ErrorException("$what: solver stopped with status $st (not OPTIMAL); the iterate is not a " *
                         "solution. Try a smaller N (or fewer quadrature nodes for QuadLogMean), " *
                         "raise the solver's iteration limit via `optimizer`, or pass check=false " *
                         "to get the partial iterate anyway."))
end

"""
    geodesic_socp(G::MarkovGraph, ρA, ρB; N=10, optimizer=Clarabel.Optimizer, silent=true, check=true) -> GeodesicSolution

Compute the discrete transport geodesic between densities `ρA` and `ρB` on `G` as a
single second-order-cone program, rather than via the
Chambolle-Pock primal-dual iteration (`discrete_transport`). Returns the squared
distance `W2 = ‖ρA - ρB‖_𝒲²`; the metric distance is `sqrt(W2)`.

The mobility `θ` is the graph's `G.mean`: `GeometricMean()` (default), `ArithmeticMean()`,
`HarmonicMean()` or `QuadLogMean(K)` (the logarithmic mean by Gauss–Legendre quadrature,
`K` power cones per edge and time step; a graph built with `LogarithmicMean()` has no
conic form and errors here, use `MarkovGraph(G; mean=QuadLogMean(8))`). See `_mean_cone!`
for the conic representations and `AdmissibleMean` for the theory.

`N` is the number of time-discretization intervals (`h = 1/N`); the returned `ρ` has
`N+1` columns and `m` has `N` columns. `optimizer` is any solver JuMP can dispatch to
that supports rotated second-order cone constraints (Clarabel by default). With `check=true`
(default) a solve that does not reach `OPTIMAL`/`ALMOST_OPTIMAL` throws rather than
returning the solver's last iterate (see `_check_solved`).
"""
function geodesic_socp(G::MarkovGraph, ρA::AbstractVector, ρB::AbstractVector;
                        N::Int=10, optimizer=Clarabel.Optimizer, silent::Bool=true, check::Bool=true)
    h = 1.0 / N

    model = Model(optimizer)
    silent && set_silent(model)

    blk = _geodesic_block!(model, G, N, h, ρA, ρB)
    @objective(model, Min, h * sum(G.κ[e] * blk.w[e, t] for t in 1:N, e in 1:length(G.E)))

    optimize!(model)
    check && _check_solved(model, "geodesic_socp")

    φ0, φ1 = _endpoint_potentials(G, blk)
    return GeodesicSolution(
        objective_value(model),
        value.(blk.ρ),
        value.(blk.m),
        value.(blk.m)[:, 1],
        φ0, φ1,
        termination_status(model),
        solve_time(model),
    )
end
