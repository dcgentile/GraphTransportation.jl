# GridComparison.jl
#
# Compare three barycenter methods on a 7×7 grid graph:
#   1. WGD via barycenter() + analysis()
#      (h=0.1, tol=1e-8, geodesic_tol=1e-10, geodesic_steps=8)
#   2. Sinkhorn with normalized squared shortest-path cost  (ε=0.01)
#   3. Sinkhorn with normalized squared diffusion-distance cost, t=diameter  (ε=0.01)
#      where diameter = max shortest-path length (ensures no degenerate diffusion distances)
#
# Reference measures: μ1 (lower-left, node 1), μ2 (center, node 25),
#                     μ3 (upper-right, node 49)
# Weights: w = [0.5, 0.3, 0.2]
#
# Coordinate recovery:
#   WGD       → GraphTransportation.analysis()
#   Sinkhorn  → simplex_regression()
#
# Two output figures:
#   grid_comparison_density.pdf  — all panels shown as densities w.r.t. u
#   grid_comparison_prob.pdf     — all panels shown as probability measures
#
# Run from src/experiments/ with --project=.

using GraphTransportation
using Optim                                     # required by simplex_regression
using CairoMakie, GraphMakie
using Graphs
using LinearAlgebra, SparseArrays, Statistics
using JLD2
using LaTeXStrings

# Include Sinkhorn.jl directly so that simplex_regression can see Optim in the
# calling scope.
include("./ExperimentUtils.jl")
include("../src/Sinkhorn.jl")

# Helper functions needed by form_diffusion_map_from_graph in ExperimentUtils.jl.
function adj_mat_to_markov_chain(adj_mat)
    row_sums = vec(sum(adj_mat, dims=2))
    return Diagonal(1.0 ./ row_sums) * adj_mat
end

function find_markov_steady_state(p)
    dim = size(p, 1)
    q = p - I(dim)
    q = hcat(q, ones(dim))
    QTQ = q * q'
    bQT = ones(dim)
    return QTQ \ bQT
end

include("./ExperimentUtils.jl")

# ── Grid graph setup ───────────────────────────────────────────────────────────
const N_GRID = 7
Q, sstate = grid_markov_chain(N_GRID)
n = N_GRID^2   # 49 nodes

adj = Float64.(Q .> 0)   # binary adjacency matrix (no self-loops)

g_plot = SimpleGraph(n)
for i in 1:n, j in i+1:n
    adj[i, j] != 0 && add_edge!(g_plot, i, j)
end

# Node i → column = ((i-1) % N_GRID) + 1, row = div(i-1, N_GRID) + 1
positions   = [Point2f(((i-1) % N_GRID) + 1, div(i-1, N_GRID) + 1) for i in 1:n]
grid_layout = (_) -> positions

# ── Node indices ───────────────────────────────────────────────────────────────
const IDX_LL  = 1   # lower-left  corner (row 1, col 1)
const IDX_CTR = 25  # center node (row 4, col 4)
const IDX_UR  = 49  # upper-right corner (row 7, col 7)

# ── Parameters ─────────────────────────────────────────────────────────────────
const WEIGHTS   = [0.5, 0.3, 0.2]
const REG       = 0.0125
const H         = 0.1
const TOL       = 1e-10
const GEO_TOL   = 1e-12
const GEO_STEPS = 3
const MAXITERS  = 8192

# ── Reference measures ─────────────────────────────────────────────────────────
# random_geographic_concentration returns a probability measure (sums to 1).
μ1 = random_geographic_concentration(adj; center=IDX_LL)
μ2 = random_geographic_concentration(adj; center=IDX_CTR)
μ3 = random_geographic_concentration(adj; center=IDX_UR)

M_prob = hcat(μ1, μ2, μ3)                            # probability measures
M_dens = hcat(μ1 ./ sstate, μ2 ./ sstate, μ3 ./ sstate)  # densities w.r.t. u

# ── Cost matrices ──────────────────────────────────────────────────────────────
println("Computing shortest-path cost..."); flush(stdout)
sp_dist = compute_graph_metric(adj)
sp_cost = sp_dist .^ 2
sp_cost ./= maximum(sp_cost)
println("  done. range=[$(minimum(sp_cost)), $(maximum(sp_cost))]"); flush(stdout)

# DIFF_T = graph diameter: smallest t such that no pair of nodes has zero diffusion distance.
const DIFF_T = Int(maximum(sp_dist))
println("Graph diameter = $DIFF_T  (setting DIFF_T = $DIFF_T)"); flush(stdout)

println("Computing diffusion cost (t=$DIFF_T)..."); flush(stdout)

println("  Step 1: building Markov chain from adjacency..."); flush(stdout)
_mc = adj_mat_to_markov_chain(Float64.(adj))
println("  done. size=$(size(_mc)), rowsum range=[$(minimum(sum(_mc, dims=2))), $(maximum(sum(_mc, dims=2)))]"); flush(stdout)

println("  Step 2: computing mchain^$DIFF_T ..."); flush(stdout)
_ts = _mc^DIFF_T
println("  done."); flush(stdout)

println("  Step 3: finding steady state..."); flush(stdout)
_ss = find_markov_steady_state(_mc)
println("  done. sum=$(_ss |> sum)  min=$(minimum(_ss))"); flush(stdout)

println("  Step 4: pairwise diffusion distances ($(n*(n-1)÷2) pairs)..."); flush(stdout)
_D = zeros(n, n)
for i in 1:n
    i % 10 == 1 && (println("    row $i / $n"); flush(stdout))
    for j in i+1:n
        v = ((_ts[i, :] .- _ts[j, :]) .^ 2) ./ _ss
        _D[i, j] = _D[j, i] = sqrt(sum(v))
    end
end
println("  done. range=[$(minimum(_D)), $(maximum(_D))]"); flush(stdout)

diff_cost = _D .^ 2
diff_cost ./= maximum(diff_cost)

# ── Computation (with per-result JLD2 caching) ────────────────────────────────
# Each result is saved immediately after computation so a killed run loses at
# most the current in-progress barycenter.
const CACHE = "grid_comparison.jld2"

# Upsert helper: write (or overwrite) specific keys without touching others.
function update_cache!(cache, pairs::Pair...)
    jldopen(cache, isfile(cache) ? "a+" : "w") do f
        for (k, v) in pairs
            haskey(f, k) && delete!(f, k)
            f[k] = v
        end
    end
end

# use the cache?
wgd_fresh_run = false
sp_fresh_run = false
diffusion_fresh_run = false

# Load whatever has already been cached.
bar_wgd  = nothing; rc_wgd  = nothing
bar_sp   = nothing; rc_sp   = nothing
bar_diff = nothing; rc_diff = nothing

if isfile(CACHE)
    println("Loading cached results from $CACHE ..."); flush(stdout)
    jldopen(CACHE, "r") do f
        haskey(f, "bar_wgd")  && (global bar_wgd  = f["bar_wgd"])
        haskey(f, "rc_wgd")   && (global rc_wgd   = f["rc_wgd"])
        haskey(f, "bar_sp")   && (global bar_sp   = f["bar_sp"])
        haskey(f, "rc_sp")    && (global rc_sp    = f["rc_sp"])
        haskey(f, "bar_diff") && (global bar_diff = f["bar_diff"])
        haskey(f, "rc_diff")  && (global rc_diff  = f["rc_diff"])
    end
end

# ── Kernel diagnostics ────────────────────────────────────────────────────────
function kernel_stats(name, C, ε)
    K = exp.(-C ./ ε)
    frac_zero = mean(K .< 1e-10)
    row_sums  = vec(sum(K, dims=2))
    println("$name (ε=$ε):")
    println("  cost range  : [$(round(minimum(C), sigdigits=4)), $(round(maximum(C), sigdigits=4))]")
    println("  kernel sparsity : $(round(frac_zero*100; sigdigits=4))%  (entries < 1e-10)")
    println("  row-sum range   : [$(round(minimum(row_sums), sigdigits=4)), $(round(maximum(row_sums), sigdigits=4))]")
    flush(stdout)
end
kernel_stats("SP cost",   sp_cost,   REG)
kernel_stats("Diff cost", diff_cost, REG)

if isnothing(bar_wgd) || isnothing(rc_wgd) || wgd_fresh_run
    println("=== WGD barycenter ==="); flush(stdout)
    bar_wgd = barycenter(M_dens, WEIGHTS, Q;
                         h=H, tol=TOL, maxiters=MAXITERS,
                         geodesic_tol=GEO_TOL, geodesic_steps=GEO_STEPS)
    rc_wgd = vec(analysis(bar_wgd, M_dens, Q; N=GEO_STEPS, tol=GEO_TOL))
    println("  WGD recovered: $(round.(rc_wgd; sigdigits=4))"); flush(stdout)
    update_cache!(CACHE, "bar_wgd" => bar_wgd, "rc_wgd" => rc_wgd)
end

if isnothing(bar_sp) || isnothing(rc_sp) || sp_fresh_run
    println("=== Sinkhorn barycenter (shortest-path cost) ==="); flush(stdout)
    bar_sp = sinkhorn_barycenter(WEIGHTS, M_prob, nothing, sp_cost, REG)
    rc_sp  = simplex_regression(M_prob, bar_sp, sp_cost, REG)
    println("  SP  recovered: $(round.(rc_sp;  sigdigits=4))"); flush(stdout)
    update_cache!(CACHE, "bar_sp" => bar_sp, "rc_sp" => rc_sp)
end

if isnothing(bar_diff) || isnothing(rc_diff) || diffusion_fresh_run
    println("=== Sinkhorn barycenter (diffusion cost, t=$DIFF_T) ==="); flush(stdout)
    bar_diff = sinkhorn_barycenter(WEIGHTS, M_prob, nothing, diff_cost, REG)
    rc_diff  = simplex_regression(M_prob, bar_diff, diff_cost, REG)
    println("  Diff recovered: $(round.(rc_diff; sigdigits=4))"); flush(stdout)
    update_cache!(CACHE, "bar_diff" => bar_diff, "rc_diff" => rc_diff)
end

# ── Coordinate summary ─────────────────────────────────────────────────────────
rel_err(rc) = norm(rc .- WEIGHTS) / norm(WEIGHTS)

println("\nTrue coordinates:        $(round.(WEIGHTS;   sigdigits=3))")
println("WGD recovered:           $(round.(rc_wgd;   sigdigits=3))" *
        "  rel_err=$(round(rel_err(rc_wgd);  sigdigits=3))")
println("Sinkhorn/SP recovered:   $(round.(rc_sp;    sigdigits=3))" *
        "  rel_err=$(round(rel_err(rc_sp);   sigdigits=3))")
println("Sinkhorn/diff recovered: $(round.(rc_diff;  sigdigits=3))" *
        "  rel_err=$(round(rel_err(rc_diff); sigdigits=3))")
flush(stdout)

# ── Shared plotting helpers ────────────────────────────────────────────────────
const CMAP      = :plasma
const NODE_SIZE = 16

# Panels use an explicit GridLayout so that rowgap! and Labels work correctly.
# Optional sublabels (LaTeXStrings) are rendered below the graph + colorbar row.
function draw_panel!(fig_pos, nc; title=L"", sublabels=LaTeXString[])
    gl = GridLayout(fig_pos)
    ax = Axis(gl[1, 1]; title=title, aspect=DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    nc_f = collect(Float64, nc)
    vmin, vmax = extrema(nc_f)
    graphplot!(ax, g_plot;
        layout=grid_layout,
        node_color=nc_f,
        node_size=NODE_SIZE,
        node_attr=(colormap=CMAP, colorrange=(vmin, vmax)),
        edge_color=(:gray60, 0.5),
        edge_width=0.75,
    )
    Colorbar(gl[1, 2]; colormap=CMAP, limits=(vmin, vmax), height=Relative(0.75))
    if !isempty(sublabels)
        for (i, lbl) in enumerate(sublabels)
            Label(gl[i+1, 1:2], lbl; tellwidth=false)
        end
        rowgap!(gl, 1, 4)
        for i in 1:length(sublabels)-1
            rowgap!(gl, i+1, 2)
        end
    end
end

# Sublabels for a barycenter panel: optional params line, true weights,
# recovered weights, and relative error.
function bary_sublabels(rc; params=nothing)
    err = round(rel_err(rc); sigdigits=3)
    lbls = LaTeXString[]
    !isnothing(params) && push!(lbls, latexstring(params))
    push!(lbls, latexstring("\\lambda = $(round.(WEIGHTS; sigdigits=2))"))
    push!(lbls, latexstring("\\hat{\\lambda} = $(round.(rc; sigdigits=3)),\\quad e_r = $err"))
    lbls
end

# ── Figure 1: densities w.r.t. u ──────────────────────────────────────────────
# References: μᵢ / u   |   Barycenters: bar_wgd (native), bar_sp/u, bar_diff/u
println("\nRendering density figure ..."); flush(stdout)

fig_dens = Figure(size=(1400, 900))

draw_panel!(fig_dens[1, 1], μ1 ./ sstate; title=L"\mu_1 / u \;\text{(lower-left, node 1)}")
draw_panel!(fig_dens[1, 2], μ2 ./ sstate; title=L"\mu_2 / u \;\text{(center, node 25)}")
draw_panel!(fig_dens[1, 3], μ3 ./ sstate; title=L"\mu_3 / u \;\text{(upper-right, node 49)}")

draw_panel!(fig_dens[2, 1], bar_wgd;
    title=L"\nu_\lambda\ \text{Wasserstein Gradient Descent}",
    sublabels=bary_sublabels(rc_wgd; params="h=$(H),\\ \\delta_g = $(TOL)"))
draw_panel!(fig_dens[2, 2], bar_sp ./ sstate;
    title=L"\nu_\lambda\ \text{Sinkhorn, Shortest Path Cost}",
    sublabels=bary_sublabels(rc_sp; params="\\varepsilon_{\\mathrm{reg}} = $(REG)"))
draw_panel!(fig_dens[2, 3], bar_diff ./ sstate;
    title=latexstring("\\nu_\\lambda\\ \\text{Sinkhorn, Diffusion Distance Cost}"),
    sublabels=bary_sublabels(rc_diff; params="t=$(DIFF_T),\\ \\varepsilon_{\\mathrm{reg}} = $(REG)"))

save("grid_comparison_density.pdf", fig_dens)
println("Saved grid_comparison_density.pdf")

# ── Figure 2: probability measures (sum to 1) ──────────────────────────────────
# References: μᵢ   |   Barycenters: bar_wgd*u, bar_sp, bar_diff
println("Rendering probability figure ..."); flush(stdout)

fig_prob = Figure(size=(1400, 900))

draw_panel!(fig_prob[1, 1], μ1; title=L"\mu_1 \;\text{(lower-left, node 1)}")
draw_panel!(fig_prob[1, 2], μ2; title=L"\mu_2 \;\text{(center, node 25)}")
draw_panel!(fig_prob[1, 3], μ3; title=L"\mu_3 \;\text{(upper-right, node 49)}")

draw_panel!(fig_prob[2, 1], bar_wgd .* sstate;
    title=L"\nu_\lambda\ \text{Wasserstein Gradient Descent}",
    sublabels=bary_sublabels(rc_wgd; params="h=$(H),\\ \\delta_g = $(TOL)"))
draw_panel!(fig_prob[2, 2], bar_sp;
    title=L"\nu_\lambda\ \text{Sinkhorn, Shortest Path Cost}",
    sublabels=bary_sublabels(rc_sp; params="\\varepsilon_{\\mathrm{reg}} = $(REG)"))
draw_panel!(fig_prob[2, 3], bar_diff;
    title=latexstring("\\nu_\\lambda\\ \\text{Sinkhorn, Diffusion Distance Cost}"),
    sublabels=bary_sublabels(rc_diff; params="t=$(DIFF_T),\\ \\varepsilon_{\\mathrm{reg}} = $(REG)"))

save("grid_comparison_prob.pdf", fig_prob)
println("Saved grid_comparison_prob.pdf")
