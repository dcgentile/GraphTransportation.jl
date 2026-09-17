# MassachusettsSOCPComparison.jl
#
# Compare the fixed-step (correct) intrinsic-descent barycenter against the
# provably-optimal SOCP barycenter (`barycenter_socp`) on the MA house graph,
# using the same reference measures/weights as MassachusettsBarycenter.jl so
# this is directly relatable to that paper figure. This is descent vs SOCP.
#
# Reports/plots:
#   - the two barycenters side by side on the real MA geography, + |difference|
#   - "variance": J_descent (evaluated via SOCP geodesics, following
#     BarycenterSOCPValidation.jl's convention) vs J_socp, plus the optimality
#     certificate J_socp <= J_descent
#   - the descent scheme's loss curve (WGD variance trace across iterations),
#     with J_socp drawn as a horizontal reference line
#   - relative error between the recovered coordinates (analysis for the descent
#     barycenter, analyze_socp for the SOCP one) against the true λ, and between
#     each other. Why the two recovery pipelines are not interchangeable between
#     methods is a separate question: see SynthesisAnalysisMismatch.jl.
#
# Run from src/experiments/ with --project=.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using LaTeXStrings
using CairoMakie, GraphMakie
using Graphs
using LinearAlgebra, SparseArrays, Statistics, Printf
using Shapefile, LibGEOS
using JLD2

# --- Same reference measures as MassachusettsBarycenter.jl ---
Q_unw, u_unw = ma_house_markov_chain()
n_nodes = size(Q_unw, 1)
A = Q_unw .> 0

n_measures = 4
weight = 10
centers = [5, 50, 100, 145]
M = zeros(n_nodes, n_measures)
for (k, c) in enumerate(centers)
    M[:, k] = random_geographic_concentration(A, weight=weight, center=c) ./ u_unw
end
λ = [0.4, 0.3, 0.2, 0.1]

# --- Parameters (matching MassachusettsBarycenter.jl's geo_tol=1e-10 production run) ---
N        = 2
h        = 0.1
tol      = 1e-8
geo_tol  = 1e-10
maxiters = 8192

norm_pi(v, π) = sqrt(sum(π .* v .^ 2))

"""
    objective_at(G, refs, λ, ν; N)

`Σᵢ λᵢ 𝒲_h²(refs[i], ν)` evaluated via `geodesic_socp`, i.e. `J_descent` in the
same normalization as `barycenter_socp`'s `J` — the convention established in
`BarycenterSOCPValidation.jl`.
"""
function objective_at(G::MarkovGraph, refs, λ, ν; N)
    sum(λ[i] * geodesic_socp(G, refs[i], ν; N=N).W2 for i in eachindex(refs))
end

CACHE = "ma_house_socp_comparison.jld2"

if isfile(CACHE)
    println("Loading cached results from $CACHE ...")
    d = load(CACHE)
    bar_descent = d["bar_descent"]; diffs = d["diffs"]; variances = d["variances"]
    ν_socp = d["ν_socp"]; J_socp = d["J_socp"]
    rc_descent = d["rc_descent"]
    t_descent = d["t_descent"]; t_socp = d["t_socp"]
    # Re-analyze ν_socp (cheap) rather than trusting the cached value, which may
    # predate the current analyze_socp default; nothing is resynthesized.
    G = MarkovGraph(Q_unw, u_unw)
    refs = [M[:, i] for i in 1:n_measures]
    rc_socp = vec(analyze_socp(G, ν_socp, refs; N=N))
else
    println("Running descent barycenter (fixed-step CP, N=$N, tol=$tol, geo_tol=$geo_tol, maxiters=$maxiters) ...")
    t_descent = @elapsed (bar_descent, diffs, variances) = barycenter(M, λ, Q_unw;
        h=h, tol=tol, geodesic_tol=geo_tol, geodesic_steps=N, maxiters=maxiters,
        geodesic_warmstart=true, return_stats=true, verbose=true)

    println("Running SOCP barycenter ...")
    G = MarkovGraph(Q_unw, u_unw)
    refs = [M[:, i] for i in 1:n_measures]
    t_socp = @elapsed (ν_socp, J_socp, geos_socp) = barycenter_socp(G, refs, λ; N=N)

    println("Recovering barycentric coordinates (analysis vs analyze_socp) ...")
    rc_descent = vec(analysis(bar_descent, M, Q_unw; N=N, tol=geo_tol))
    rc_socp    = vec(analyze_socp(G, ν_socp, refs; N=N))

    @save CACHE bar_descent diffs variances ν_socp J_socp rc_descent rc_socp t_descent t_socp
end

G = MarkovGraph(Q_unw, u_unw)
refs = [M[:, i] for i in 1:n_measures]

J_descent     = objective_at(G, refs, λ, bar_descent; N=N)
discrepancy   = norm_pi(bar_descent .- ν_socp, u_unw)
certificate_ok = J_socp <= J_descent + 1e-6 * max(abs(J_descent), 1.0)

wgd_iters = something(findlast(!=(0.0), variances), length(variances))

rc_err_descent = norm(λ .- rc_descent) / norm(λ)
rc_err_socp    = norm(λ .- rc_socp) / norm(λ)
rc_diff        = norm(rc_descent .- rc_socp) / norm(rc_descent)

println("== MA house: descent vs SOCP ==")
@printf("  wall-clock: descent=%.1fs  socp=%.3fs\n", t_descent, t_socp)
@printf("  ‖ν_descent - ν_socp‖_π = %.6g\n", discrepancy)
@printf("  J_socp = %.6g   J_descent (via SOCP geodesics) = %.6g   certificate (J_socp <= J_descent): %s\n",
        J_socp, J_descent, certificate_ok)
println("  -- coordinate recovery --")
println("    true λ:              ", λ)
@printf("    recovered (descent): %s   rel. err %.4e\n", round.(rc_descent; sigdigits=4), rc_err_descent)
@printf("    recovered (SOCP):    %s   rel. err %.4e\n", round.(rc_socp; sigdigits=4), rc_err_socp)
@printf("    ‖rc_descent - rc_socp‖ / ‖rc_descent‖ = %.4e\n", rc_diff)

# --- Geographic positions from shapefile centroids (same as MassachusettsBarycenter.jl) ---
GI = LibGEOS.GeoInterface
shp_file = joinpath(@__DIR__, "HOUSE2021", "HOUSE2021_POLY.shp")
table = Shapefile.Table(shp_file)
geoms = Shapefile.shapes(table)
positions = map(geoms) do geom
    g = GI.convert(LibGEOS.MultiPolygon, geom)
    c = LibGEOS.centroid(g)
    Point2f(GI.getcoord(c, 1), GI.getcoord(c, 2))
end

g_plot = SimpleGraph(n_nodes)
for i in 1:n_nodes, j in i+1:n_nodes
    A[i, j] != 0 && add_edge!(g_plot, i, j)
end

abs_diff = abs.(bar_descent .- ν_socp)
crange = (min(minimum(bar_descent), minimum(ν_socp)), max(maximum(bar_descent), maximum(ν_socp)))
crange = (crange[2] - crange[1]) < 1e-4 * max(abs(crange[1]), abs(crange[2]), 1.0) ?
          (crange[1] - 1, crange[2] + 1) : crange
diff_range = (0.0, max(maximum(abs_diff), 1e-6))

fig = Figure(size=(1300, 950))

ax1 = Axis(fig[1, 1], title="descent (fixed-step, correct)", aspect=DataAspect())
ax2 = Axis(fig[1, 2], title="SOCP (globally optimal)", aspect=DataAspect())
ax3 = Axis(fig[1, 3], title="|ν_descent − ν_SOCP|", aspect=DataAspect())
hidedecorations!(ax1); hidedecorations!(ax2); hidedecorations!(ax3)
hidespines!(ax1); hidespines!(ax2); hidespines!(ax3)

graphplot!(ax1, g_plot; layout=positions, node_color=bar_descent, colormap=:viridis,
           colorrange=crange, node_size=12, edge_color=(:gray, 0.2))
graphplot!(ax2, g_plot; layout=positions, node_color=ν_socp, colormap=:viridis,
           colorrange=crange, node_size=12, edge_color=(:gray, 0.2))
graphplot!(ax3, g_plot; layout=positions, node_color=abs_diff, colormap=:inferno,
           colorrange=diff_range, node_size=12, edge_color=(:gray, 0.2))

Colorbar(fig[2, 1:2], colormap=:viridis, colorrange=crange,
          label="density (w.r.t. steady state)", vertical=false)
Colorbar(fig[2, 3], colormap=:inferno, colorrange=diff_range,
          label="|difference|", vertical=false)

ax4 = Axis(fig[3, 1:3], title="Descent loss curve (WGD variance per iterate) vs J_SOCP",
           xlabel="iteration", ylabel="variance (Σ λᵢ W²(ν, Mᵢ))", yscale=log10)
lines!(ax4, 1:wgd_iters, variances[1:wgd_iters], label="descent variance")
hlines!(ax4, [J_socp]; color=:red, linestyle=:dash, label="J_SOCP (optimal)")
axislegend(ax4)

Label(fig[0, :],
      @sprintf("MA house: descent vs SOCP   —   ‖ν_descent−ν_SOCP‖_π=%.4e   J_descent=%.6g  J_SOCP=%.6g  (certificate: %s)",
                discrepancy, J_descent, J_socp, certificate_ok),
      fontsize=15, tellwidth=false)
Label(fig[4, :],
      @sprintf("coordinate recovery — true λ=%s | descent: %s (rel.err %.4e) | SOCP: %s (rel.err %.4e) | ‖Δrc‖/‖rc_descent‖=%.4e",
                round.(λ; sigdigits=4), round.(rc_descent; sigdigits=4), rc_err_descent,
                round.(rc_socp; sigdigits=4), rc_err_socp, rc_diff),
      fontsize=12, tellwidth=false)

save("ma_house_socp_comparison.pdf", fig)
println("Saved ma_house_socp_comparison.{jld2,pdf}")
