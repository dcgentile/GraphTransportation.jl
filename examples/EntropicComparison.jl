# EntropicComparison.jl
#
# Compare barycenter methods on the USA state adjacency graph:
#   [WGD disabled — see commented block below]
#   1. Sinkhorn barycenter with normalized squared shortest-path cost
#   2. Sinkhorn barycenter with normalized squared diffusion-distance cost,
#      for t ∈ {2, 4, 8, 16}
#
# Reference measures are geographically concentrated near:
#   California (node 15), Maine (node 37), Tennessee (node 22)
# with barycentric weights (0.5, 0.3, 0.2).
#
# Coordinate recovery: simplex_regression()
#
# Run from src/experiments/ so that ./data/states.shp resolves correctly.

using GraphTransportation
using Optim                                     # required by simplex_regression
using CairoMakie, GraphMakie
using Graphs
using Shapefile, LibGEOS
using LinearAlgebra, SparseArrays, Statistics
using JLD2

# Include Sinkhorn.jl directly so that simplex_regression can see Optim in the
# calling scope. (GraphTransportation exports these names too; the include
# shadows them with identical definitions that have Optim available.)
include("../src/Sinkhorn.jl")

# Helper functions needed by form_diffusion_map_from_graph in ExperimentUtils.jl.
# These are simple random-walk utilities not exported from GraphTransportation;
# the definitions here follow the pattern in experiments/SAMPTA.jl.
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

# ── USA graph setup ────────────────────────────────────────────────────────────
# load_usa_mc() reads ./data/states.shp; must be run from src/experiments/.
Qusa, sstate, geo_cx, geo_cy = load_usa_mc()
n = length(sstate)

adj = Float64.(Qusa .> 0)          # binary adjacency matrix, no self-loops

g_plot = SimpleGraph(n)
for i in 1:n, j in i+1:n
    adj[i, j] != 0 && add_edge!(g_plot, i, j)
end

positions  = [Point2f(geo_cx[i], geo_cy[i]) for i in 1:n]
geo_layout = (_) -> positions

# ── Reference measures ─────────────────────────────────────────────────────────
# Node indices in states.shp (ordering determined empirically; c.f.
# experiments/SAMPTA.jl::barycenter_experiment and Initializations.jl comments).
const IDX_CA = 15   # California
const IDX_ME = 37   # Maine
const IDX_TN = 22   # Tennessee

const WEIGHTS    = [0.5, 0.3, 0.2]
const REG        = 0.01              # Sinkhorn regularization parameter
const DIFF_TIMES = [2, 4, 8, 16]    # diffusion time parameters (t=2 under investigation)

# Probability measures (each sums to 1 by L1 norm).
μ_CA = random_geographic_concentration(adj; center=IDX_CA)
μ_ME = random_geographic_concentration(adj; center=IDX_ME)
μ_TN = random_geographic_concentration(adj; center=IDX_TN)

M_prob = hcat(μ_CA, μ_ME, μ_TN)
# M_dens = hcat(μ_CA ./ sstate, μ_ME ./ sstate, μ_TN ./ sstate) # WGD densities (disabled)

# ── Cost matrices ──────────────────────────────────────────────────────────────
flush(stdout)
println("Computing cost matrices...")

# All cost matrices are normalized to [0, 1] before use.  Without
# normalization, the kernel exp(-C/ε) has ~89% underflow entries for
# squared shortest-path distances at ε=0.125, and similarly for squared
# diffusion distances at small t.  Normalization keeps the max kernel
# exponent at -1/ε while preserving the relative structure of the cost.
sp_cost = compute_graph_metric(adj) .^ 2
sp_cost ./= maximum(sp_cost)

diff_costs = map(DIFF_TIMES) do t
    D = form_diffusion_map_from_graph(adj, t) .^ 2
    D ./ maximum(D)
end

# ── Kernel diagnostics ─────────────────────────────────────────────────────────
# At REG=0.01 the kernel K = exp(-C/ε) decays fast; sparsity is the fraction of
# entries that underflow to < 1e-10.  High sparsity → poor Sinkhorn conditioning
# and unreliable simplex_regression gradients.
println("\n=== Kernel diagnostics (ε=$REG) ===")
flush(stdout)
function kernel_stats(name, C)
    K = exp.(-C ./ REG)
    row_sums = vec(sum(K, dims=2))
    frac_zero = mean(K .< 1e-10)
    println("$name:")
    println("  cost  range : [$(round(minimum(C), sigdigits=4)), $(round(maximum(C), sigdigits=4))]")
    println("  kernel sparsity : $(round(frac_zero*100; sigdigits=4))%  (entries < 1e-10)")
    println("  kernel row-sum  : min=$(round(minimum(row_sums), sigdigits=4))  max=$(round(maximum(row_sums), sigdigits=4))  mean=$(round(mean(row_sums), sigdigits=4))")
    flush(stdout)
    return K
end
K_sp = kernel_stats("Shortest-path (normalized)", sp_cost)
K_diffs = map(enumerate(DIFF_TIMES)) do (i, t)
    kernel_stats("Diffusion t=$t (normalized)", diff_costs[i])
end
println()

# Local override of simplex_regression with iteration cap and convergence tracing.
# The default (no Options) allows 1000 LBFGS iterations; each calls loss_gradient
# which runs 2048 Sinkhorn steps.  At ε=0.01 the kernel is nearly singular so
# gradients may be NaN/Inf — cap at 200 iterations and print the result summary.
function simplex_regression_traced(measures, target, cost, epsilon; maxiters=200)
    num_measures = size(measures, 2)
    x0 = fill(1.0 / num_measures, num_measures)
    f(coords)    = barycentric_loss(coords, measures, target, cost, epsilon)
    function g!(G, coords)
        grad = loss_gradient(coords, measures, cost, target, epsilon)
        copyto!(G, grad)
    end
    opts   = Optim.Options(iterations=maxiters, show_trace=false)
    result = Optim.optimize(f, g!, x0, Optim.LBFGS(), opts)
    println("    Optim: $(Optim.summary(result))  converged=$(Optim.converged(result))  iters=$(result.iterations)  f=$(round(Optim.minimum(result); sigdigits=4))  |g|=$(round(Optim.g_residual(result); sigdigits=4))")
    flush(stdout)
    return logarithmic_change_of_variable(Optim.minimizer(result))
end

# ── Computation (with JLD2 caching) ───────────────────────────────────────────
const CACHE = "entropic_comparison.jld2"

if isfile(CACHE)
    println("Loading cached results from $CACHE ...")
    @load CACHE bar_sp rc_sp bars_diff rcs_diff
else
    # === WGD barycenter (disabled) ===
    # steps = 10
    # bar_wgd = barycenter(M_dens, WEIGHTS, Qusa;
    #                      h=0.05, maxiters=1000, tol=1e-8,
    #                      geodesic_tol=1e-7, geodesic_steps=steps)
    # rc_wgd = vec(analysis(bar_wgd, M_dens, Qusa; N=steps))

    println("=== Sinkhorn barycenter (squared shortest-path cost) ==="); flush(stdout)
    bar_sp = sinkhorn_barycenter(WEIGHTS, M_prob, nothing, sp_cost, REG)
    println("  barycenter sum = $(sum(bar_sp))  min = $(minimum(bar_sp))"); flush(stdout)
    rc_sp  = simplex_regression_traced(M_prob, bar_sp, sp_cost, REG)
    println("  → recovered: $(round.(rc_sp; sigdigits=4))"); flush(stdout)

    bars_diff = zeros(n, length(DIFF_TIMES))
    rcs_diff  = zeros(3, length(DIFF_TIMES))
    for (i, t) in enumerate(DIFF_TIMES)
        println("=== Sinkhorn barycenter (diffusion cost, t=$t) ==="); flush(stdout)
        bars_diff[:, i] = sinkhorn_barycenter(WEIGHTS, M_prob, nothing, diff_costs[i], REG)
        println("  barycenter sum = $(sum(bars_diff[:, i]))  min = $(minimum(bars_diff[:, i]))"); flush(stdout)
        println("  Attempting simplex_regression (t=$t)..."); flush(stdout)
        try
            rcs_diff[:, i] = simplex_regression_traced(M_prob, bars_diff[:, i], diff_costs[i], REG)
            println("  → recovered: $(round.(rcs_diff[:, i]; sigdigits=4))"); flush(stdout)
        catch e
            println("  !! simplex_regression FAILED (t=$t): $e"); flush(stdout)
            rcs_diff[:, i] .= NaN
        end
    end

    @save CACHE bar_sp rc_sp bars_diff rcs_diff
end

println("\nTrue coordinates:      $(round.(WEIGHTS; sigdigits=3))")
println("Sinkhorn/SP recovered: $(round.(rc_sp;   sigdigits=3))")
for (i, t) in enumerate(DIFF_TIMES)
    println("Sinkhorn/diff(t=$t) recovered: $(round.(rcs_diff[:, i]; sigdigits=3))")
end
flush(stdout)

# ── Visualization ──────────────────────────────────────────────────────────────
# Layout (3 × 3):
#   row 1: CA | ME | TN
#   row 2: SP | diff(t=2) | diff(t=4)
#   row 3: diff(t=8) | diff(t=16) | (empty)
#
# WGD panel disabled — re-enable once WGD experiment is restored.

cmap      = :magma
node_size = 10

function draw_panel!(pos, nc; title="")
    ax = Axis(pos[1, 1]; title=title, aspect=DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    nc_f = collect(Float64, nc)
    vmin, vmax = extrema(nc_f)
    graphplot!(ax, g_plot;
        layout=geo_layout,
        node_color=nc_f,
        node_size=node_size,
        node_attr=(colormap=cmap, colorrange=(vmin, vmax)),
        edge_color=(:gray60, 0.5),
        edge_width=0.75,
    )
    Colorbar(pos[1, 2]; colormap=cmap, limits=(vmin, vmax), height=Relative(0.75))
end

fig = Figure(size=(1400, 900))

draw_panel!(fig[1, 1], μ_CA; title="California  (node $IDX_CA)")
draw_panel!(fig[1, 2], μ_ME; title="Maine  (node $IDX_ME)")
draw_panel!(fig[1, 3], μ_TN; title="Tennessee  (node $IDX_TN)")

draw_panel!(fig[2, 1], bar_sp;
    title="Sinkhorn / shortest-path  (ε=$REG)\n" *
          "true: $(round.(WEIGHTS; sigdigits=2))\n" *
          "recovered: $(round.(rc_sp; sigdigits=3))")

draw_panel!(fig[2, 2], bars_diff[:, 1];
    title="Sinkhorn / diffusion  (t=$(DIFF_TIMES[1]), ε=$REG)\n" *
          "true: $(round.(WEIGHTS; sigdigits=2))\n" *
          "recovered: $(round.(rcs_diff[:, 1]; sigdigits=3))")

draw_panel!(fig[2, 3], bars_diff[:, 2];
    title="Sinkhorn / diffusion  (t=$(DIFF_TIMES[2]), ε=$REG)\n" *
          "true: $(round.(WEIGHTS; sigdigits=2))\n" *
          "recovered: $(round.(rcs_diff[:, 2]; sigdigits=3))")

draw_panel!(fig[3, 1], bars_diff[:, 3];
    title="Sinkhorn / diffusion  (t=$(DIFF_TIMES[3]), ε=$REG)\n" *
          "true: $(round.(WEIGHTS; sigdigits=2))\n" *
          "recovered: $(round.(rcs_diff[:, 3]; sigdigits=3))")

draw_panel!(fig[3, 2], bars_diff[:, 4];
    title="Sinkhorn / diffusion  (t=$(DIFF_TIMES[4]), ε=$REG)\n" *
          "true: $(round.(WEIGHTS; sigdigits=2))\n" *
          "recovered: $(round.(rcs_diff[:, 4]; sigdigits=3))")

save("entropic_comparison.pdf", fig)
println("Saved entropic_comparison.pdf")
