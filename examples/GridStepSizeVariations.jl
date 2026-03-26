using GraphTransportation
using GraphMakie, Graphs, CairoMakie
using JLD2, Printf
using LinearAlgebra, SparseArrays
include("./ExperimentUtils.jl")

# --- Setup: n×n grid graph ---
grid_size = 4
Q, u = grid_markov_chain(grid_size)
A = Q .> 0
n_nodes = size(Q, 1)

weight = 10
centers = [1, 4, 13]   # top-left, top-right, bottom-left corners of the 4×4 grid
M = zeros(n_nodes, length(centers))
for i in eachindex(centers)
    M[:,i] = random_geographic_concentration(A, weight=weight, center=centers[i]) ./ u
end

coord    = [1/3; 1/3; 1/3]
h        = 0.1
maxiters = 1024
tol      = 1e-10
geo_tol  = 1e-6

# Variants to benchmark:
#   cold_steps  : cold-start barycenter at each individual geodesic step count
#   iterated_*  : warm-started iterated_barycenter with two different schedules
cold_steps      = [2, 4, 8, 16, 32]
schedule_fine   = [2, 4, 8, 16, 32]
schedule_coarse = [2, 16, 32]

shared_kwargs = (h=h, maxiters=maxiters, tol=tol, geodesic_tol=geo_tol, verbose=false)

# --- Compute (or load) all outputs ---
cache = "stepsize_variations.jld2"
if isfile(cache)
    d             = load(cache)
    cold_bars     = d["cold_bars"]
    iter_fine     = d["iter_fine"]
    iter_coarse   = d["iter_coarse"]
    cold_times    = d["cold_times"]
    time_fine     = d["time_fine"]
    time_coarse   = d["time_coarse"]
else
    cold_bars   = Vector{Vector{Float64}}(undef, length(cold_steps))
    cold_times  = Vector{Float64}(undef, length(cold_steps))
    for (i, gs) in enumerate(cold_steps)
        println("Cold start: geodesic_steps=$gs")
        cold_times[i] = @elapsed cold_bars[i] = barycenter(M, coord, Q;
                             geodesic_steps=gs, shared_kwargs...)
    end

    println("Iterated (fine schedule: $schedule_fine)")
    time_fine   = @elapsed iter_fine   = iterated_barycenter(M, coord, Q;
                                             steps=schedule_fine,   shared_kwargs...)

    println("Iterated (coarse schedule: $schedule_coarse)")
    time_coarse = @elapsed iter_coarse = iterated_barycenter(M, coord, Q;
                                             steps=schedule_coarse, shared_kwargs...)

    @save cache cold_bars cold_times iter_fine iter_coarse time_fine time_coarse
end

# --- Timing report ---
println("\n=== Timing ===")
for (i, gs) in enumerate(cold_steps)
    @printf("  cold  geodesic_steps=%2d : %.3f s\n", gs, cold_times[i])
end
@printf("  iterated fine   %s : %.3f s\n", string(schedule_fine),   time_fine)
@printf("  iterated coarse %s : %.3f s\n", string(schedule_coarse), time_coarse)

# --- Quality: pairwise u-norm distances ---
all_labels = [@sprintf("cold(%d)", gs) for gs in cold_steps]
push!(all_labels, "iter_fine", "iter_coarse")
all_bars = vcat(cold_bars, [iter_fine, iter_coarse])

println("\n=== Pairwise u-norm distances ===")
for i in eachindex(all_bars), j in (i+1):length(all_bars)
    d = norm((all_bars[i] - all_bars[j]) .* sqrt.(u))
    @printf("  %-16s vs %-16s : %.6f\n", all_labels[i], all_labels[j], d)
end

# --- Graph structure for plotting ---
g_plot = SimpleGraph(n_nodes)
for i in 1:n_nodes, j in i+1:n_nodes
    A[i, j] != 0 && add_edge!(g_plot, i, j)
end

grid_positions = [Point2f((k-1) % grid_size, (k-1) ÷ grid_size) for k in 1:n_nodes]
grid_layout = (_) -> grid_positions

# --- Figure: 2 rows ---
# Row 1: input measures (n_measures panels)
# Row 2: all barycenter outputs (cold per step + 2 iterated)
M_disp    = M .* u
bars_disp = [b .* u for b in all_bars]

gmin, gmax = extrema(vcat(vec(M_disp), vcat(bars_disp...)))
cmap = :magma

n_measures = size(M, 2)
n_bars     = length(all_bars)
ncols      = max(n_measures, n_bars)

fig = Figure(size=(280*ncols + 60, 560))

function add_graph_panel!(row, col, data, title)
    ax = Axis(fig[row, col], title=title, titlesize=10)
    hidedecorations!(ax); hidespines!(ax)
    graphplot!(ax, g_plot;
        layout=grid_layout,
        node_color=collect(Float64, data),
        node_size=20,
        node_attr=(colormap=cmap, colorrange=(gmin, gmax)),
        edge_color=(:gray60, 0.4),
        edge_width=2,
    )
end

# Top row: input measures
for i in 1:n_measures
    add_graph_panel!(1, i, M_disp[:,i], "ν$(i)")
end

# Bottom row: all barycenter outputs
for (i, (lbl, t)) in enumerate(zip(all_labels, vcat(cold_times, [time_fine, time_coarse])))
    add_graph_panel!(2, i, bars_disp[i], "$lbl\n$(round(t; sigdigits=3))s")
end

Colorbar(fig[1:2, ncols+1]; colormap=cmap, limits=(gmin, gmax))

save("stepsize_variations.pdf", fig)
println("\nSaved stepsize_variations.pdf")
