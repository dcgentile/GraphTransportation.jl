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
    n = G.n
    nE = length(G.E)
    h = 1.0 / N

    model = Model(optimizer)
    silent && set_silent(model)

    @variable(model, ρ[1:n, 1:(N+1)] >= 0)
    @variable(model, m[1:nE, 1:N])
    @variable(model, ϑ[1:nE, 1:N] >= 0)
    @variable(model, w[1:nE, 1:N] >= 0)

    @constraint(model, ρ[:, 1] .== ρA)
    @constraint(model, ρ[:, N+1] .== ρB)

    # discrete continuity equation: (ρ_{i+1} - ρ_i)/h + div(m_i) = 0
    for i in 1:N
        divm = graph_divergence(G, m[:, i])
        @constraint(model, (ρ[:, i+1] .- ρ[:, i]) ./ h .+ divm .== 0)
    end

    # mean cone (rotated SOC): ϑ_{e,i}² ≤ ρ̄_{x,i} ρ̄_{y,i}
    # action epigraph (rotated SOC): m_{e,i}² ≤ ϑ_{e,i} w_{e,i}
    for i in 1:N, (e, (x, y)) in enumerate(G.E)
        ρ̄x = (ρ[x, i] + ρ[x, i+1]) / 2
        ρ̄y = (ρ[y, i] + ρ[y, i+1]) / 2
        @constraint(model, [ρ̄x, ρ̄y, sqrt(2) * ϑ[e, i]] in RotatedSecondOrderCone())
        @constraint(model, [ϑ[e, i], w[e, i], sqrt(2) * m[e, i]] in RotatedSecondOrderCone())
    end

    @objective(model, Min, h * sum(G.κ[e] * w[e, i] for i in 1:N, e in 1:nE))

    optimize!(model)

    return GeodesicSolution(
        objective_value(model),
        value.(ρ),
        value.(m),
        value.(m)[:, 1],
        termination_status(model),
        solve_time(model),
    )
end
