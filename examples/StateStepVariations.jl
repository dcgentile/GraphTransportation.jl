using CairoMakie, GraphMakie, Graphs
using Printf
using LinearAlgebra, SparseArrays, Statistics
using JLD2, Random
using GraphTransportation
include("./ExperimentUtils.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment: StateStepVariations
#
# Goal: investigate how the number of geodesic steps (N) used during barycenter
# *synthesis* interacts with the N used during *analysis* (coordinate recovery).
#
# Hypothesis: `barycenter` and `analysis` both approximate geodesics via the
# same discrete-transport solver.  When N_synthesis == N_analysis the two
# approximations are *consistent* — the analysed barycenter lives at the same
# point in the approximate geometry as the one that was synthesised — so the
# recovered coordinates should be closest to the true weights.  Using a finer
# N for analysis than for synthesis may help marginally if the true barycenter
# lies at a fine-grid optimum, but using a coarser N almost certainly hurts
# because the analysis geodesics then point in subtly wrong directions.
#
# We test this by fixing three reference measures and one weight vector, then
# synthesising a barycenter at each N in SYN_STEPS, and for each synthesised
# result running analysis at every N in AN_STEPS.  This produces a 3×3 table
# of relative coordinate-recovery errors whose diagonal is the matched case.
#
# Setup
#   Reference measures : Oregon (44), Arizona (34), Illinois (28)
#   Weights            : random, seed 42
#   Synthesis params   : h=0.1, tol=1e-9, geo_tol=1e-11, maxiters=2^16
#                        geo_steps ∈ {4, 16, 32}
#   Analysis N         : {4, 16, 32} for each synthesised barycenter
#
# Run from src/experiments/ with:
#   julia --project=. StateStepVariations.jl
# ─────────────────────────────────────────────────────────────────────────────

const CACHE_FILE  = "state_step_variations.jld2"
const FIG_PLAIN   = "state_step_variations.pdf"
const FIG_TABLE   = "state_step_variations_table.pdf"

const CENTERS      = [44, 34, 28]          # Oregon, Arizona, Illinois
const CENTER_NAMES = ["Oregon", "Arizona", "Illinois"]
const SYN_STEPS    = [4, 16, 32]
const AN_STEPS     = [4, 16, 32]

const H        = 0.1
const TOL      = 1e-9
const GEO_TOL  = 1e-11
const MAXITERS = 2^16
const SEED     = 42
const WEIGHT   = 8    # geographic-concentration weight for reference measures


function run_experiment()
    Q, u, geo_cx, geo_cy = load_usa_mc()
    A = Q .> 0
    M = hcat([random_geographic_concentration(A, weight=WEIGHT, center=c) ./ u
              for c in CENTERS]...)

    Random.seed!(SEED)
    weights = rand(length(CENTERS))
    weights ./= sum(weights)
    println("Weights (seed=$SEED): ", round.(weights; digits=4))

    # Cache stores:
    #   "bary_gs$(N)"          → Vector{Float64}  (synthesised barycenter)
    #   "err_gs$(syn)_an$(an)" → Float64           (relative coord-recovery error)
    # The cache is written after every individual computation so that a killed
    # run can resume from exactly where it left off.
    cache      = isfile(CACHE_FILE) ? load(CACHE_FILE, "cache") : Dict{String,Any}()
    cache_lock = ReentrantLock()

    nthreads = Threads.nthreads()
    println("Using $nthreads thread(s)")

    # ── Phase 1: synthesis ────────────────────────────────────────────────────
    # The three barycenters are independent and can run in parallel.
    # verbose=true only in single-threaded mode; interleaved progress bars from
    # multiple threads would be unreadable.
    barycenters  = Vector{Vector{Float64}}(undef, length(SYN_STEPS))
    syn_tasks    = collect(enumerate(SYN_STEPS))
    n_syn        = length(syn_tasks)
    syn_done     = Threads.Atomic{Int}(0)

    function run_synthesis!(si, gs; verbose=false)
        key = "bary_gs$(gs)"
        cached = lock(cache_lock) do
            get(cache, key, nothing)
        end
        if !isnothing(cached)
            k = Threads.atomic_add!(syn_done, 1) + 1
            println("[$k/$n_syn] Cache hit: barycenter gs=$gs  [t$(Threads.threadid())]")
            barycenters[si] = cached
        else
            println("[  /$n_syn] Synthesising barycenter  gs=$gs  [t$(Threads.threadid())]")
            bary = barycenter(M, weights, Q;
                              h=H, tol=TOL, maxiters=MAXITERS,
                              geodesic_steps=gs, geodesic_tol=GEO_TOL,
                              verbose=verbose)
            barycenters[si] = bary          # unique index per task — no lock needed
            k = Threads.atomic_add!(syn_done, 1) + 1
            println("[$k/$n_syn] Done: barycenter gs=$gs  [t$(Threads.threadid())]  → cached")
            lock(cache_lock) do
                cache[key] = bary
                @save CACHE_FILE cache
            end
        end
    end

    if nthreads == 1
        for (si, gs) in syn_tasks
            run_synthesis!(si, gs; verbose=true)
        end
    else
        Threads.@threads for (si, gs) in syn_tasks
            run_synthesis!(si, gs)
        end
    end

    # ── Phase 2: analysis ─────────────────────────────────────────────────────
    # err_mat[i, j] = ‖w - ŵ‖/‖w‖  for synthesis with SYN_STEPS[i]
    #                               and analysis  with  AN_STEPS[j].
    # The diagonal (i == j) is the matched case we expect to perform best.
    # All 9 tasks are independent given the barycenters above.
    err_mat   = zeros(length(SYN_STEPS), length(AN_STEPS))
    an_tasks  = vec([(si, gs, ai, an)
                     for (si, gs) in enumerate(SYN_STEPS),
                         (ai, an) in enumerate(AN_STEPS)])
    n_an      = length(an_tasks)
    an_done   = Threads.Atomic{Int}(0)

    function run_analysis!(si, gs, ai, an)
        key = "err_gs$(gs)_an$(an)"
        cached = lock(cache_lock) do
            get(cache, key, nothing)
        end
        if !isnothing(cached)
            k = Threads.atomic_add!(an_done, 1) + 1
            println("[$k/$n_an] Cache hit: analysis  gs=$gs  an=$an  →  $(@sprintf("%.4e", cached))  [t$(Threads.threadid())]")
            err_mat[si, ai] = cached
        else
            println("[  /$n_an] Analysing: synthesis gs=$gs, analysis N=$an  [t$(Threads.threadid())]")
            rcs = vec(analysis(barycenters[si], M, Q; N=an, tol=GEO_TOL))
            err = norm(weights .- rcs) / norm(weights)
            err_mat[si, ai] = err           # unique index per task — no lock needed
            k = Threads.atomic_add!(an_done, 1) + 1
            println("[$k/$n_an] Done: analysis  gs=$gs  an=$an  →  $(@sprintf("%.4e", err))  [t$(Threads.threadid())]")
            lock(cache_lock) do
                cache[key] = err
                @save CACHE_FILE cache
            end
        end
    end

    if nthreads == 1
        for (si, gs, ai, an) in an_tasks
            run_analysis!(si, gs, ai, an)
        end
    else
        Threads.@threads for (si, gs, ai, an) in an_tasks
            run_analysis!(si, gs, ai, an)
        end
    end

    return barycenters, err_mat, M, weights, Q, u, geo_cx, geo_cy
end


function build_graph(Q)
    A = Q .> 0
    n = size(Q, 1)
    g = SimpleGraph(n)
    for i in 1:n, j in i+1:n
        A[i, j] != 0 && add_edge!(g, i, j)
    end
    return g
end


"""
    add_measure_panels!(fig, M, barycenters, g, positions; ref_row, bary_row, cmap)

Populate `fig` with reference measures (row `ref_row`) and synthesised
barycenters (row `bary_row`).  Returns `((mmin,mmax), (bmin,bmax))` for
downstream colorbar construction.
"""
function add_measure_panels!(fig, M, barycenters, g, positions;
                             ref_row=1, bary_row=2, cmap=:magma)
    mmin, mmax = extrema(M)
    bmin, bmax = extrema(reduce(vcat, barycenters))

    for (i, name) in enumerate(CENTER_NAMES)
        ax = Axis(fig[ref_row, i]; title=name)
        hidedecorations!(ax); hidespines!(ax)
        graphplot!(ax, g;
            layout     = (_) -> positions,
            node_color = M[:, i],
            node_size  = 20,
            node_attr  = (colormap=cmap, colorrange=(mmin, mmax)),
            edge_color = :black)
    end

    for (si, gs) in enumerate(SYN_STEPS)
        ax = Axis(fig[bary_row, si]; title="Barycenter  (N=$gs)")
        hidedecorations!(ax); hidespines!(ax)
        graphplot!(ax, g;
            layout     = (_) -> positions,
            node_color = barycenters[si],
            node_size  = 20,
            node_attr  = (colormap=cmap, colorrange=(bmin, bmax)),
            edge_color = :black)
    end

    return (mmin, mmax), (bmin, bmax)
end


function plot_plain(M, barycenters, g, positions)
    ncols = length(CENTERS)
    fig   = Figure(size=(300*ncols + 80, 620))

    (mmin, mmax), (bmin, bmax) =
        add_measure_panels!(fig, M, barycenters, g, positions; ref_row=1, bary_row=2)

    Label(fig[1, 0], "References";  rotation=π/2, tellheight=false)
    Label(fig[2, 0], "Barycenters"; rotation=π/2, tellheight=false)
    Colorbar(fig[1, ncols+1]; colormap=:magma, limits=(mmin, mmax))
    Colorbar(fig[2, ncols+1]; colormap=:magma, limits=(bmin, bmax))

    return fig
end


function plot_with_table(M, barycenters, err_mat, g, positions)
    ncols = length(CENTERS)
    n     = length(SYN_STEPS)
    fig   = Figure(size=(300*ncols + 80, 950))

    (mmin, mmax), (bmin, bmax) =
        add_measure_panels!(fig, M, barycenters, g, positions; ref_row=1, bary_row=2)

    Label(fig[1, 0], "References";  rotation=π/2, tellheight=false)
    Label(fig[2, 0], "Barycenters"; rotation=π/2, tellheight=false)
    Colorbar(fig[1, ncols+1]; colormap=:magma, limits=(mmin, mmax))
    Colorbar(fig[2, ncols+1]; colormap=:magma, limits=(bmin, bmax))

    # Error heatmap.
    # heatmap!(ax, Z): first index of Z → x-axis, second → y-axis.
    # err_mat[i, j]: i = synthesis-step index (x), j = analysis-step index (y).
    log_err = log10.(err_mat)
    ax_err  = Axis(fig[3, 1:n];
        title   = "Coordinate recovery error  ‖w − ŵ‖/‖w‖  (log₁₀)",
        xlabel  = "Synthesis N",
        ylabel  = "Analysis N",
        xticks  = (1:n, string.(SYN_STEPS)),
        yticks  = (1:n, string.(AN_STEPS)),
        aspect  = n)
    rowsize!(fig.layout, 3, Relative(0.22))

    hm  = heatmap!(ax_err, log_err; colormap=:viridis)
    mid = (minimum(log_err) + maximum(log_err)) / 2
    for i in 1:n, j in 1:n
        tc = log_err[i, j] < mid ? :white : :black
        text!(ax_err, @sprintf("%.2e", err_mat[i, j]);
            position = (i, j), align = (:center, :center),
            color = tc, fontsize = 10)
    end
    Colorbar(fig[3, n+1], hm; label="log₁₀(error)")

    return fig
end


# ── Entry point ───────────────────────────────────────────────────────────────
barycenters, err_mat, M, weights, Q, u, geo_cx, geo_cy = run_experiment()

# Terminal error table
println("\n── Coordinate recovery errors  ‖w − ŵ‖/‖w‖ ──────────────────────────")
println(@sprintf("%-14s", "syn╲an"), join([@sprintf("%12s", "an=$an") for an in AN_STEPS]))
for (si, gs) in enumerate(SYN_STEPS)
    row = join([@sprintf("%12.4e", err_mat[si, ai]) for ai in eachindex(AN_STEPS)])
    println(@sprintf("syn=%-10d", gs), row)
end
println()

g         = build_graph(Q)
positions = [Point2f(geo_cx[i], geo_cy[i]) for i in 1:size(Q, 1)]

fig_plain = plot_plain(M, barycenters, g, positions)
save(FIG_PLAIN, fig_plain)
println("Saved $FIG_PLAIN")

fig_table = plot_with_table(M, barycenters, err_mat, g, positions)
save(FIG_TABLE, fig_table)
println("Saved $FIG_TABLE")
