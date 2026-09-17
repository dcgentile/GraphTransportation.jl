# bench_logexp.jl
#
# exp/log-map round-trip errors and Newton iteration histograms for log_map/exp_map,
# plus wall-clock and W2 agreement against geodesic_socp, on
#   - the 5x5 grid (25 nodes) and the 10x10 grid (100 nodes), random interior pairs
#   - the MA house graph (160 nodes), random interior pairs
# Also reports warm-started iteration counts (shoot to a perturbed target
# starting from the previous φ0).
#
# Run from src/experiments/ with --project=.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using LinearAlgebra, Random, Statistics, Printf
using Shapefile, LibGEOS
using CairoMakie
using JLD2

const NTRIALS = 20
const N_SOCP  = 100

function random_density(rng, π; spread=0.5)
    v = rand(rng, length(π)) .+ spread
    v ./ dot(v, π)
end

cases = Dict{String,Any}()
Q5, π5 = grid_markov_chain(5);   cases["grid 5x5 (25)"]   = MarkovGraph(Q5, π5)
Q10, π10 = grid_markov_chain(10); cases["grid 10x10 (100)"] = MarkovGraph(Q10, π10)
Qma, πma = ma_house_markov_chain(); cases["MA house (160)"] = MarkovGraph(Qma, πma)

# JIT warm-up
let G = cases["grid 5x5 (25)"], rng = MersenneTwister(0)
    ν = random_density(rng, G.π); μ = random_density(rng, G.π)
    r = log_map(G, ν, μ); exp_map(G, ν, r.φ0); geodesic_socp(G, ν, μ; N=10)
end

const CACHE = "bench_logexp.jld2"
if isfile(CACHE)
    results = load(CACHE, "results")
else
    results = Dict{String,NamedTuple}()
    for name in ["grid 5x5 (25)", "grid 10x10 (100)", "MA house (160)"]
        G = cases[name]
        rng = MersenneTwister(11)
        iters = Int[]; iters_warm = Int[]; roundtrip = Float64[]; t_log = Float64[]; t_socp = Float64[]
        W2_rel = Float64[]; m0_rel = Float64[]
        println("== $name ==")
        for k in 1:NTRIALS
            ν = random_density(rng, G.π); μ = random_density(rng, G.π)
            tl = @elapsed r = log_map(G, ν, μ)
            push!(t_log, tl); push!(iters, r.iters)
            push!(roundtrip, norm((exp_map(G, ν, r.φ0) .- μ) .* sqrt.(G.π)))
            ts = @elapsed sol = geodesic_socp(G, ν, μ; N=N_SOCP)
            push!(t_socp, ts)
            push!(W2_rel, abs(r.W2 - sol.W2) / sol.W2)
            push!(m0_rel, norm(r.m0 .- sol.m0) / norm(sol.m0))
            # warm start: nudge the target slightly and reuse φ0
            μ2 = 0.95 .* μ .+ 0.05 .* random_density(rng, G.π)
            push!(iters_warm, log_map(G, ν, μ2; φ0_init=r.φ0).iters)
            @printf("  trial %2d: iters=%d warm=%d roundtrip=%.1e W2rel=%.1e m0rel=%.1e  log_map %.2fs  socp(N=%d) %.2fs\n",
                    k, r.iters, iters_warm[end], roundtrip[end], W2_rel[end], m0_rel[end], tl, N_SOCP, ts)
        end
        results[name] = (; iters, iters_warm, roundtrip, t_log, t_socp, W2_rel, m0_rel)
    end
    @save CACHE results
end

println("\n== Summary ==")
@printf("%-18s %-12s %-12s %-10s %-10s %-10s %-10s %-10s\n", "graph", "iters cold", "iters warm", "roundtrip", "W2 rel", "m0 rel", "t_log(s)", "t_socp(s)")
for name in ["grid 5x5 (25)", "grid 10x10 (100)", "MA house (160)"]
    r = results[name]
    @printf("%-18s %-12s %-12s %-10.1e %-10.1e %-10.1e %-10.3f %-10.3f\n", name,
            "$(minimum(r.iters))-$(maximum(r.iters)) (med $(median(r.iters)))",
            "$(minimum(r.iters_warm))-$(maximum(r.iters_warm))",
            maximum(r.roundtrip), maximum(r.W2_rel), maximum(r.m0_rel), median(r.t_log), median(r.t_socp))
end

fig = Figure(size=(1200, 400))
for (col, name) in enumerate(["grid 5x5 (25)", "grid 10x10 (100)", "MA house (160)"])
    r = results[name]
    ax = Axis(fig[1, col], title=name, xlabel="Newton iterations", ylabel="count")
    hist!(ax, r.iters; bins=0:maximum(r.iters)+1, label="cold", color=(:steelblue, 0.7))
    hist!(ax, r.iters_warm; bins=0:maximum(r.iters)+1, label="warm", color=(:orange, 0.6))
    axislegend(ax)
end
Label(fig[0, :], "log_map Newton iteration counts (random interior pairs, $NTRIALS trials each)", fontsize=15, tellwidth=false)
save("bench_logexp.pdf", fig)
println("Saved bench_logexp.{jld2,pdf}")
