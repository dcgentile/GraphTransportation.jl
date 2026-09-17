"""
    barycenter_socp(G, refs, λ; N=10, optimizer=Clarabel.Optimizer, silent=true)
        -> (ν::Vector{Float64}, J::Float64, geodesics::Vector{GeodesicSolution})

Compute the discrete transport barycenter of reference measures `refs` (a vector of
probability densities on `G`, one per reference) with weights `λ` as a single joint
second-order-cone program, rather than by intrinsic gradient
descent (`barycenter`). Replicates the geodesic SOCP block (`_geodesic_block!`)
once per reference with `λ[i] > 0`, sharing a single free right endpoint `ν` across
all of them.

Returns:
- `ν`: the barycenter, i.e. the shared optimal right endpoint.
- `J`: the optimal objective value, `Σᵢ λᵢ 𝒲_h²(refs[i], ν)`.
- `geodesics`: one `GeodesicSolution` per reference with `λ[i] > 0` (in the same
  relative order as `refs`), giving the geodesic from that reference to `ν`. Each
  carries its endpoint potentials (`φ0` at the reference, `φ1` at `ν`, both for the
  *unweighted* geodesic — the `λ[i]` factor is divided out of the duals). KKT
  stationarity of the joint program with respect to `ν` is exactly
  `Σᵢ λᵢ φ1ᵢ = const` on the support of `ν`, which is the stationarity condition the
  potential-based analysis (`analyze_socp`) checks.

References with `λ[i] == 0` are dropped entirely rather than solved with a zero
weight. The mobility is `G.mean`, as in `geodesic_socp`. `λ` must be a probability vector (`λ .>= 0`, `sum(λ) ≈ 1`).
"""
function barycenter_socp(G::MarkovGraph, refs::Vector{<:AbstractVector}, λ::AbstractVector;
                          N::Int=10, optimizer=Clarabel.Optimizer, silent::Bool=true)
    @assert length(refs) == length(λ)
    @assert all(λ .>= 0)
    @assert sum(λ) ≈ 1.0 atol=1e-8

    active = findall(>(0), λ)
    @assert !isempty(active) "at least one λ[i] must be > 0"

    nE = length(G.E)
    h = 1.0 / N

    model = Model(optimizer)
    silent && set_silent(model)

    @variable(model, ν[1:G.n] >= 0)
    @constraint(model, dot(ν, G.π) == 1)

    blocks = [(ref_index=i, _geodesic_block!(model, G, N, h, refs[i], ν; base_name=string(i))...)
              for i in active]

    @objective(model, Min,
        h * sum(λ[b.ref_index] * G.κ[e] * b.w[e, t] for b in blocks for t in 1:N, e in 1:nE))

    optimize!(model)

    status = termination_status(model)
    st = solve_time(model)

    geodesics = map(blocks) do b
        # This block's action enters the objective weighted by λ[i], so its endpoint
        # duals are λ[i] times the unweighted geodesic's; divide that back out.
        φ0, φ1 = _endpoint_potentials(G, b; weight=λ[b.ref_index])
        GeodesicSolution(
            h * sum(G.κ[e] * value(b.w[e, t]) for t in 1:N, e in 1:nE),
            value.(b.ρ),
            value.(b.m),
            value.(b.m)[:, 1],
            φ0, φ1,
            status,
            st,
        )
    end

    return value.(ν), objective_value(model), geodesics
end
