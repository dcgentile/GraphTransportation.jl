# boundary_stress.jl
#
# Measures supported on subsets of nodes. Two questions:
#  (1) SOCP geodesics/barycenters with Dirac-type (single-node) references on a cycle and
#      a grid: does the solver handle zero-density nodes cleanly, and how coarse can N be?
#      (Barycenters of Diracs turn out to be fully supported at every N, so the plot shows
#      the geodesic's W2 vs N instead: Dirac endpoints need larger N than interior ones.)
#  (2) Mollified shooting for a boundary-supported target: error of the extrapolated
#      and raw-smallest-ε distances vs. the SOCP reference as a function of ε, i.e.
#      does the mollification error decay like √ε (the model behind log_map_mollified's
#      fit) or faster?
#
# Run from src/experiments/ with --project=.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using LinearAlgebra, Random, Statistics, Printf
using CairoMakie
using JLD2

dirac(G, x) = (v = zeros(G.n); v[x] = 1 / G.π[x]; v)

const CACHE = "boundary_stress.jld2"

function compute_boundary_stress()
    # ---- (1) Diracs on a 6-cycle and a 5x5 grid ----
    part1 = Dict{String,Any}()
    for (name, (Q, π), a, b, refs_idx) in (("6-cycle", markov_chain_from_edge_list([(1,2),(2,3),(3,4),(4,5),(5,6),(6,1)]), 1, 4, [1, 3, 5]),
                                            ("5x5 grid", grid_markov_chain(5), 1, 25, [1, 5, 21, 25]))
        G = MarkovGraph(Q, π)
        println("== $name ==")
        geo = Dict{Int,Any}()
        for N in (2, 5, 10, 20, 40)
            g = geodesic(G, dirac(G, a), dirac(G, b); N=N)
            support = count(>(1e-6), g.ρ[:, N ÷ 2 + 1])
            geo[N] = (; W2=g.W2, status=g.status, midpoint_support=support, t=g.solvetime)
            @printf("  Dirac geodesic N=%-3d W2=%.5f status=%s midpoint support=%d nodes  (%.2fs)\n", N, g.W2, g.status, support, g.solvetime)
        end
        refs = [dirac(G, i) for i in refs_idx]; λ = fill(1 / length(refs), length(refs))
        bar = Dict{Int,Any}()
        for N in (2, 5, 10, 20)
            t = @elapsed (ν, J, _) = barycenter(G, refs, λ; N=N)
            bar[N] = (; ν, J, support=count(>(1e-6), ν), t)
            @printf("  Dirac barycenter N=%-3d J=%.5f support=%d nodes min=%.1e  (%.2fs)\n", N, J, count(>(1e-6), ν), minimum(ν), t)
        end
        part1[name] = (; geo, bar)
    end

    # ---- (2) mollified shooting vs ε on a 5x5 grid, boundary-supported target ----
    Q, π = grid_markov_chain(5); G = MarkovGraph(Q, π)
    rng = MersenneTwister(7)
    ν = rand(rng, G.n) .+ 0.5; ν ./= dot(ν, π)
    tgt = zeros(G.n); for i in 1:G.n; mod(i - 1, 5) < 2 && (tgt[i] = 1.0 + rand(rng)); end; tgt ./= dot(tgt, π)
    W_ref = sqrt(geodesic(G, ν, tgt; N=800).W2)
    εs = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    function mollified_sweep(G, ν, tgt, εs)
        Ws = Float64[]; used = Float64[]; r = nothing
        for ε in εs
            try
                r = log_map(G, (1 - ε) .* ν .+ ε, (1 - ε) .* tgt .+ ε; φ0_init=(r === nothing ? nothing : r.φ0), tol=1e-7)
                push!(Ws, sqrt(r.W2)); push!(used, ε)
                @printf("  ε=%.0e  W=%.6f  |W-W_ref|=%.2e  iters=%d\n", ε, sqrt(r.W2), abs(sqrt(r.W2) - W_ref), r.iters)
            catch e
                println("  ε=$ε failed: ", first(sprint(showerror, e), 120))
            end
        end
        return Ws, used
    end
    Ws, used = mollified_sweep(G, ν, tgt, εs)
    fit = log_map_mollified(G, ν, tgt)
    @printf("  W_ref=%.6f  √ε-fit extrapolation=%.6f (err %.2e)  raw smallest-ε=%.6f (err %.2e)\n",
            W_ref, fit.W, abs(fit.W - W_ref), fit.Ws[end], abs(fit.Ws[end] - W_ref))
    # local decay exponent between successive ε levels
    slopes = [log(abs(Ws[i] - W_ref) / abs(Ws[i+1] - W_ref)) / log(used[i] / used[i+1]) for i in 1:length(used)-1]
    part2 = (; εs=used, Ws, W_ref, fit_W=fit.W, fit_raw=fit.Ws[end], slopes)
    println("  local decay exponents p in |W(ε)-W_ref| ~ ε^p: ", round.(slopes, digits=2), "  (√ε model predicts 0.5)")
    return part1, part2
end

if isfile(CACHE)
    d = load(CACHE); part1 = d["part1"]; part2 = d["part2"]
else
    part1, part2 = compute_boundary_stress()
    @save CACHE part1 part2
end

fig = Figure(size=(1100, 450))
ax1 = Axis(fig[1, 1], title="Dirac-to-Dirac SOCP geodesic: W2 vs N", xlabel="N", ylabel="W2 (relative to N=40)")
for (name, p) in part1
    Ns = sort(collect(keys(p.geo)))
    scatterlines!(ax1, Ns, [p.geo[N].W2 / p.geo[Ns[end]].W2 for N in Ns]; label=name)
end
hlines!(ax1, [1.0]; color=:gray, linestyle=:dash)
axislegend(ax1; position=:rb)
ax2 = Axis(fig[1, 2], title="Mollified shooting: |W(ε) − W_ref| vs ε", xlabel="ε", ylabel="abs. error", xscale=log10, yscale=log10)
scatterlines!(ax2, part2.εs, abs.(part2.Ws .- part2.W_ref); label="raw W(ε)")
lines!(ax2, part2.εs, abs(part2.Ws[1] - part2.W_ref) .* sqrt.(part2.εs ./ part2.εs[1]); linestyle=:dash, color=:gray, label="√ε reference")
hlines!(ax2, [abs(part2.fit_W - part2.W_ref)]; color=:red, linestyle=:dot, label="√ε-fit extrapolation error")
axislegend(ax2; position=:lt)
save("boundary_stress.pdf", fig)
println("Saved boundary_stress.{jld2,pdf}")
