using CairoMakie, Printf
using LinearAlgebra, SparseArrays, Statistics
using JLD2, Random
using GraphTransportation
include("./ExperimentUtils.jl")

const CACHE_FILE = "param_variations_usa.jld2"
const FIGURE_FILE = "param_variations_usa.pdf"

"""
    run_experiment(; geo_tols, geo_steps, h, tol, maxiters, seed)

For each (geo_steps, geo_tol) combination, compute the WGD barycenter of five
geographic reference measures on the USA graph, then run `analysis` to recover
barycentric coordinates and record the relative error ‖w - ŵ‖ / ‖w‖.

Results are cached in `CACHE_FILE`; already-computed cells are skipped.
Returns `(err_mat, geo_steps, geo_tols)` where `err_mat[i,j]` is the error for
`geo_steps[i]` and `geo_tols[j]`.
"""
function run_experiment(;
        geo_tols  = [1e-8, 1e-9, 1e-10],
        geo_steps = [8, 16, 32, 64, 128],
        h         = 0.05,
        tol       = 1e-8,
        maxiters  = 2^16,
        seed      = 42)

    Q, u = load_usa_mc()
    A = Q .> 0
    centers = [44, 34, 28]#, 4, 8]   # Oregon, Arizona, Illinois, Virginia, Massachusetts
    M = hcat([random_geographic_concentration(A, weight=8, center=c) ./ u for c in centers]...)

    Random.seed!(seed)
    weights = rand(size(M, 2))
    weights ./= sum(weights)

    # Load or initialise cache
    cache      = isfile(CACHE_FILE) ? load(CACHE_FILE, "cache") : Dict{String,Float64}()
    cache_lock = ReentrantLock()

    nthreads = Threads.nthreads()
    println("Using $nthreads thread(s)")

    err_mat = zeros(length(geo_steps), length(geo_tols))

    # Flatten to a single task list so @threads can distribute freely
    tasks = vec([(i, gs, j, gt)
                 for (i, gs) in enumerate(geo_steps),
                     (j, gt) in enumerate(geo_tols)])
    N = length(tasks)
    done = Threads.Atomic{Int}(0)

    function run_task!(i, gs, j, gt; verbose=false)
        key = "gs$(gs)_gt$(gt)"

        cached_val = lock(cache_lock) do
            get(cache, key, nothing)
        end

        if !isnothing(cached_val)
            k = Threads.atomic_add!(done, 1) + 1
            println("[$k/$N] Cache hit  [t$(Threads.threadid())]:  gs=$gs  gt=$gt  →  err=$(cached_val)")
            err_mat[i, j] = cached_val
        else
            println("[  /  ] Computing  [t$(Threads.threadid())]:  gs=$gs  gt=$gt")
            cell_tol = max(tol, gt)
            bary = barycenter(M, weights, Q;
                              h=h, tol=cell_tol, maxiters=maxiters,
                              geodesic_steps=gs, geodesic_tol=gt,
                              verbose=verbose)
            rcs  = vec(analysis(bary, M, Q, N=gs))
            err  = norm(weights .- rcs) / norm(weights)
            k = Threads.atomic_add!(done, 1) + 1
            println("[$k/$N] Done       [t$(Threads.threadid())]:  gs=$gs  gt=$gt  →  err=$(@sprintf("%.4e", err))")
            err_mat[i, j] = err
            lock(cache_lock) do
                cache[key] = err
                @save CACHE_FILE cache
            end
        end
    end

    if nthreads == 1
        for (i, gs, j, gt) in tasks
            run_task!(i, gs, j, gt; verbose=true)
        end
    else
        Threads.@threads for (i, gs, j, gt) in tasks
            run_task!(i, gs, j, gt)
        end
    end

    return err_mat, geo_steps, geo_tols
end

"""
    plot_heatmap(err_mat, geo_steps, geo_tols)

Produce a heatmap of `log10.(err_mat)` with annotated cell values and save
it to `FIGURE_FILE`.  Returns the CairoMakie figure.
"""
function plot_heatmap(err_mat, geo_steps, geo_tols)
    log_err_mat = log10.(err_mat)

    f  = Figure(size=(800, 600))
    ax = Axis(f[1, 1];
        xticks    = (1:length(geo_steps), string.(geo_steps)),
        yticks    = (1:length(geo_tols),  string.(geo_tols)),
        xlabel    = "Steps in Geodesic",
        ylabel    = "Convergence Threshold for Geodesics",
        title     = "Coordinate Recovery Error (Log₁₀ scale)",
        yreversed = true)

    hm = heatmap!(ax, log_err_mat; colormap=:viridis)

    mid = (minimum(log_err_mat) + maximum(log_err_mat)) / 2
    for i in axes(err_mat, 1), j in axes(err_mat, 2)
        text_color = log_err_mat[i, j] < mid ? :white : :black
        text!(ax, @sprintf("%.2e", err_mat[i, j]);
            position  = (i, j),
            align     = (:center, :center),
            color     = text_color,
            fontsize  = 10)
    end

    Colorbar(f[1, 2], hm;
        label = "Error (log₁₀ scale)",
        ticks = round.(range(minimum(log_err_mat), maximum(log_err_mat); length=5); digits=1))

    save(FIGURE_FILE, f)
    return f
end

# ── entry point ──────────────────────────────────────────────────────────────
# Launch with:  julia --threads auto --project=. ParameterVariationsUSA.jl
err_mat, geo_steps, geo_tols = run_experiment()
f = plot_heatmap(err_mat, geo_steps, geo_tols)
display(f)
