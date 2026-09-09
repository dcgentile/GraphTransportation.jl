"""
    GeodesicSolution

Result of solving Module 1's geodesic SOCP (`geodesic_socp`). Note the naming
convention: `W2` is the **squared** discrete transport distance; the metric
itself is `sqrt(W2)`.

Fields:
- `W2::Float64`: squared discrete transport distance, i.e. the SOCP objective value.
- `ρ::Matrix{Float64}`: the `n × (N+1)` density path, `ρ[:, i+1]` at time `i/N`.
- `m::Matrix{Float64}`: the `|E| × N` momentum path.
- `m0::Vector{Float64}`: `m[:, 1]`, the initial momentum (convenience).
- `status`: JuMP termination status of the solve.
- `solvetime::Float64`: wall-clock solve time in seconds.
"""
struct GeodesicSolution
    W2::Float64
    ρ::Matrix{Float64}
    m::Matrix{Float64}
    m0::Vector{Float64}
    status::Any
    solvetime::Float64
end

"""
    _geodesic_block!(model, G, N, h, left, right; base_name="") -> (; ρ, m, ϑ, w)

Add one geodesic's worth of variables and constraints (Module 1.1) to `model`:
density/momentum/mean/action variables, the discrete continuity equation, and the
mean-cone and action-epigraph RSOC constraints. `left` and `right` fix the two
boundary densities and may each be either a plain vector (as in `geodesic_socp`) or
themselves JuMP variables/expressions (as in `barycenter_socp`, where `right` is the
shared barycenter variable `ν`) — `@constraint(model, ρ[:, k] .== x)` accepts either.
Does not set an objective; callers combine one or more blocks' `w` fields into theirs.
"""
function _geodesic_block!(model, G::MarkovGraph, N::Int, h::Float64,
                           left::AbstractVector, right::AbstractVector; base_name::String="")
    n = G.n
    nE = length(G.E)

    ρ = @variable(model, [1:n, 1:(N+1)], lower_bound = 0, base_name = "ρ" * base_name)
    m = @variable(model, [1:nE, 1:N], base_name = "m" * base_name)
    ϑ = @variable(model, [1:nE, 1:N], lower_bound = 0, base_name = "ϑ" * base_name)
    w = @variable(model, [1:nE, 1:N], lower_bound = 0, base_name = "w" * base_name)

    @constraint(model, ρ[:, 1] .== left)
    @constraint(model, ρ[:, N+1] .== right)

    # discrete continuity equation: (ρ_{t+1} - ρ_t)/h + div(m_t) = 0
    for t in 1:N
        divm = graph_divergence(G, m[:, t])
        @constraint(model, (ρ[:, t+1] .- ρ[:, t]) ./ h .+ divm .== 0)
    end

    # mean cone (rotated SOC): ϑ_{e,t}² ≤ ρ̄_{x,t} ρ̄_{y,t}
    # action epigraph (rotated SOC): m_{e,t}² ≤ ϑ_{e,t} w_{e,t}
    for t in 1:N, (e, (x, y)) in enumerate(G.E)
        ρ̄x = (ρ[x, t] + ρ[x, t+1]) / 2
        ρ̄y = (ρ[y, t] + ρ[y, t+1]) / 2
        @constraint(model, [ρ̄x, ρ̄y, sqrt(2) * ϑ[e, t]] in RotatedSecondOrderCone())
        @constraint(model, [ϑ[e, t], w[e, t], sqrt(2) * m[e, t]] in RotatedSecondOrderCone())
    end

    return (ρ=ρ, m=m, ϑ=ϑ, w=w)
end

"""
    geodesic_socp(G::MarkovGraph, ρA, ρB; N=10, optimizer=Clarabel.Optimizer, silent=true) -> GeodesicSolution

Compute the discrete transport geodesic between densities `ρA` and `ρB` on `G` as a
single second-order-cone program (Module 1, `spec.txt`), rather than via the
Chambolle-Pock primal-dual iteration (`discrete_transport`). Returns the squared
distance `W2 = ‖ρA - ρB‖_𝒲²`; the metric distance is `sqrt(W2)`.

Uses the geometric mean `θ(s,t) = √(st)` (see `spec.txt`); this is not configurable.

`N` is the number of time-discretization intervals (`h = 1/N`); the returned `ρ` has
`N+1` columns and `m` has `N` columns. `optimizer` is any solver JuMP can dispatch to
that supports rotated second-order cone constraints (Clarabel by default).
"""
function geodesic_socp(G::MarkovGraph, ρA::AbstractVector, ρB::AbstractVector;
                        N::Int=10, optimizer=Clarabel.Optimizer, silent::Bool=true)
    h = 1.0 / N

    model = Model(optimizer)
    silent && set_silent(model)

    blk = _geodesic_block!(model, G, N, h, ρA, ρB)
    @objective(model, Min, h * sum(G.κ[e] * blk.w[e, t] for t in 1:N, e in 1:length(G.E)))

    optimize!(model)

    return GeodesicSolution(
        objective_value(model),
        value.(blk.ρ),
        value.(blk.m),
        value.(blk.m)[:, 1],
        termination_status(model),
        solve_time(model),
    )
end
