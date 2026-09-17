# bench_geodesics.jl
#
# Geodesic solvers side by side: accuracy vs. time resolution (log-log, expect first
# order in h for the SOCP and Chambolle-Pock; fourth order for the RK4 shooting flow)
# and a wall-clock table, on three graphs of increasing size. The reference value for
# each pair is the shooting solution at a fine step (nsteps=1200), which is exact in
# time up to RK4 error and validated against the two-node closed form in the tests.
#
#   graphs:   5x5 grid (25), 10x10 grid (100), MA house (160); random interior pairs
#   methods:  :socp at N ∈ {2,5,10,20,40,80}; :chambolle_pock at N ∈ {2,5,10,20} on the
#             two grids only (it is the paper's reference solver and by far the slowest;
#             the MA house CP runs alone take hours); :shooting at nsteps ∈ {10,20,40,80,150}
#   reports:  relative W2 error and relative m0 error vs. the reference, per method and
#             resolution; median wall-clock per solve
#
# Run from src/experiments/ with --project=.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using LinearAlgebra, Random, Statistics, Printf
using Shapefile, LibGEOS
using CairoMakie
using JLD2

const NPAIRS   = 5
const N_SOCP   = [2, 5, 10, 20, 40, 80]
const N_CP     = [2, 5, 10, 20]      # CP is O(1/N) but slow; enough points to show the rate
const NSTEPS   = [10, 20, 40, 80, 150]
const NSTEPS_REF = 1200

function random_density(rng, π; spread=0.5)
    v = rand(rng, length(π)) .+ spread
    v ./ dot(v, π)
end

graphs = [("grid 5x5 (25)",   MarkovGraph(grid_markov_chain(5)...)),
          ("grid 10x10 (100)", MarkovGraph(grid_markov_chain(10)...)),
          ("MA house (160)",   MarkovGraph(ma_house_markov_chain()...))]

# JIT warm-up
let G = graphs[1][2], rng = MersenneTwister(0)
    a = random_density(rng, G.π); b = random_density(rng, G.π)
    geodesic(G, a, b; N=2); geodesic(G, a, b; method=:shooting, nsteps=10)
    geodesic(G, a, b; method=:chambolle_pock, N=2, maxiters=100)
end

const CACHE = "bench_geodesics.jld2"
if isfile(CACHE)
    results = load(CACHE, "results")
else
    results = Dict{String,Any}()
    for (name, G) in graphs
        println("== $name ==")
        rng = MersenneTwister(21)
        pairs = [(random_density(rng, G.π), random_density(rng, G.π)) for _ in 1:NPAIRS]
        refs  = [geodesic(G, a, b; method=:shooting, nsteps=NSTEPS_REF) for (a, b) in pairs]
        rec = Dict{Tuple{Symbol,Int},NamedTuple}()
        for (method, res_list, kw) in ((:socp, N_SOCP, :N), (:chambolle_pock, N_CP, :N), (:shooting, NSTEPS, :nsteps))
            method == :chambolle_pock && startswith(name, "MA") && continue
            for r in res_list
                errW = Float64[]; errm = Float64[]; times = Float64[]
                for (k, (a, b)) in enumerate(pairs)
                    extra = method == :chambolle_pock ? (; tol=1e-9, maxiters=2^16) : (;)
                    t = @elapsed g = geodesic(G, a, b; method=method, kw => r, extra...)
                    push!(times, t)
                    push!(errW, abs(g.W2 - refs[k].W2) / refs[k].W2)
                    push!(errm, norm(g.m0 .- refs[k].m0) / norm(refs[k].m0))
                end
                rec[(method, r)] = (; errW=median(errW), errm=median(errm), t=median(times))
                @printf("  %-15s %s=%-5d W2 rel.err=%.2e  m0 rel.err=%.2e  t=%.3fs\n", method, kw, r, median(errW), median(errm), median(times))
            end
        end
        results[name] = rec
    end
    @save CACHE results
end

println("\n== wall-clock (median s) at the coarsest/finest resolution ==")
@printf("%-18s %-22s %-22s %-22s\n", "graph", "socp N=2 / 80", "chambolle_pock N=2 / $(N_CP[end])", "shooting nsteps=10 / 150")
for (name, _) in graphs
    r = results[name]
    @printf("%-18s %-22s %-22s %-22s\n", name,
        @sprintf("%.3f / %.2f", r[(:socp,2)].t, r[(:socp,80)].t),
        haskey(r, (:chambolle_pock,2)) ? @sprintf("%.3f / %.2f", r[(:chambolle_pock,2)].t, r[(:chambolle_pock,N_CP[end])].t) : "n/a",
        @sprintf("%.3f / %.3f", r[(:shooting,10)].t, r[(:shooting,150)].t))
end

fig = Figure(size=(1500, 900))
for (col, (name, _)) in enumerate(graphs)
    r = results[name]
    for (row, (field, label)) in enumerate(((:errW, "relative W2 error"), (:errm, "relative m0 error")))
        ax = Axis(fig[row, col], title=(row == 1 ? name : ""), xlabel="time step h", ylabel=label,
                  xscale=log10, yscale=log10)
        for (method, res_list, mk, clr) in ((:socp, N_SOCP, :circle, :steelblue), (:chambolle_pock, N_CP, :utriangle, :darkorange), (:shooting, NSTEPS, :rect, :seagreen))
            haskey(r, (method, res_list[1])) || continue
            hs = 1 ./ res_list
            vals = [max(getfield(r[(method, n)], field), 1e-16) for n in res_list]
            scatterlines!(ax, hs, vals; label=string(method), marker=mk, color=clr)
        end
        hs = 1 ./ N_SOCP
        lines!(ax, hs, hs .* (getfield(r[(:socp, N_SOCP[1])], field) / hs[1]); color=:gray, linestyle=:dash, label="O(h)")
        row == 1 && col == 1 && axislegend(ax; position=:rb)
    end
end
Label(fig[0, :], "Geodesic solvers: error vs. time step (reference: shooting at nsteps=$NSTEPS_REF), median over $NPAIRS random interior pairs",
      fontsize=15, tellwidth=false)
save("bench_geodesics.pdf", fig)
println("Saved bench_geodesics.{jld2,pdf}")
