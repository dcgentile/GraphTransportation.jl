include("./ExperimentUtils.jl")
using GraphTransportation
using LaTeXStrings
using CairoMakie, GraphMakie
using Graphs
using LinearAlgebra, SparseArrays, Statistics
using Shapefile, LibGEOS
using JLD2

# --- Unweighted Markov chain ---
Q_unw, u_unw = ma_house_markov_chain()
n_nodes = size(Q_unw, 1)
A = Q_unw .> 0

# --- Reference measures (each set built relative to its own steady state) ---
n_measures = 4
weight = 10
centers = [5, 50, 100, 145]

λ = [0.4, 0.3, 0.2, 0.1]

# --- Barycenter parameters ---
h        = 0.1
steps    = 2
tol      = 1e-8
maxiters = 8192
const CACHE    = "ma_house_barycenter.jld2"         # geo_tol = 1e-10
const CACHE_6  = "ma_house_barycenter_gt1e-6.jld2"  # geo_tol = 1e-6

# --- Load or compute geo_tol = 1e-10 barycenter ---
geo_tol = 1e-10
tol6    = 1e-5   # convergence threshold matched to geo_tol = 1e-6 noise floor

if isfile(CACHE)
    @load CACHE M bar_unw rc_unw
    converged_unw = true
    jldopen(CACHE, "r") do f
        haskey(f, "converged_unw") && (global converged_unw = f["converged_unw"])
    end

    if !converged_unw
        bar_unw, diffs_unw, _ = barycenter(M, λ, Q_unw, h=h, tol=tol, geodesic_tol=geo_tol, geodesic_steps=steps, maxiters=maxiters, geodesic_warmstart=true, initialization=bar_unw, return_stats=true)
        last_diff = diffs_unw[findlast(!=(0.0), diffs_unw)]
        converged_unw = last_diff < tol
        rc_unw = analysis(bar_unw, M, Q_unw, N=steps)[:,1]
        @save CACHE M bar_unw rc_unw converged_unw
    end
else
    M = zeros(n_nodes, n_measures)
    for (k, c) in enumerate(centers)
        μ = random_geographic_concentration(A, weight=weight, center=c)
        M[:, k] = μ ./ u_unw
    end

    bar_unw, diffs_unw, _ = barycenter(M, λ, Q_unw, h=h, tol=tol, geodesic_tol=geo_tol, geodesic_steps=steps, maxiters=maxiters, geodesic_warmstart=true, return_stats=true)
    last_diff = diffs_unw[findlast(!=(0.0), diffs_unw)]
    converged_unw = last_diff < tol
    rc_unw = analysis(bar_unw, M, Q_unw, N=steps)[:,1]
    @save CACHE M bar_unw rc_unw converged_unw
end

# --- Load or compute geo_tol = 1e-6 barycenter ---
geo_tol6 = 1e-6

if isfile(CACHE_6)
    @load CACHE_6 bar_unw6 rc_unw6
    converged_unw6 = true
    jldopen(CACHE_6, "r") do f
        haskey(f, "converged_unw6") && (global converged_unw6 = f["converged_unw6"])
    end

    if isnothing(converged_unw6)
        bar_unw6, diffs_unw6, _ = barycenter(M, λ, Q_unw, h=0.01, tol=tol6, geodesic_tol=geo_tol6, geodesic_steps=steps, maxiters=maxiters, geodesic_warmstart=true, initialization=bar_unw6, return_stats=true)
        last_diff6 = diffs_unw6[findlast(!=(0.0), diffs_unw6)]
        converged_unw6 = last_diff6 < tol6
        rc_unw6 = analysis(bar_unw6, M, Q_unw, N=steps)[:,1]
        @save CACHE_6 bar_unw6 rc_unw6 converged_unw6
    end
else
    # Warm-start from geo_tol=1e-10 result
    bar_unw6, diffs_unw6, _ = barycenter(M, λ, Q_unw, h=h, tol=tol6, geodesic_tol=geo_tol6, geodesic_steps=steps, maxiters=maxiters, geodesic_warmstart=true, initialization=bar_unw, return_stats=true)
    last_diff6 = diffs_unw6[findlast(!=(0.0), diffs_unw6)]
    converged_unw6 = last_diff6 < tol6
    rc_unw6 = analysis(bar_unw6, M, Q_unw, N=steps)[:,1]
    @save CACHE_6 bar_unw6 rc_unw6 converged_unw6
end

# --- Geographic positions from shapefile centroids ---
GI = LibGEOS.GeoInterface
shp_file = joinpath(@__DIR__, "HOUSE2021", "HOUSE2021_POLY.shp")
table = Shapefile.Table(shp_file)
geoms = Shapefile.shapes(table)
positions = map(geoms) do geom
    g = GI.convert(LibGEOS.MultiPolygon, geom)
    c = LibGEOS.centroid(g)
    Point2f(GI.getcoord(c, 1), GI.getcoord(c, 2))
end
geo_layout = (_) -> positions

# --- SimpleGraph for plotting ---
g_plot = SimpleGraph(n_nodes)
for i in 1:n_nodes, j in i+1:n_nodes
    A[i, j] != 0 && add_edge!(g_plot, i, j)
end

# --- Plot ---
# Left: 4 reference measures in 2x2 grid; Right: two barycenters stacked vertically
cmap = :plasma
fig = Figure(size=(1200, 800))

node_size = 16
edge_width = 2

titles = [L"\nu_1", L"\nu_2", L"\nu_3", L"\nu_4"]
panel_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for (idx, (nc, title)) in enumerate(zip(eachcol(M), titles))
    r, c = panel_positions[idx]
    pos = fig[r, c]
    ax = Axis(pos[1, 1], title=title, aspect=DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    vmin, vmax = extrema(nc)
    graphplot!(ax, g_plot;
        layout=geo_layout,
        node_color=collect(nc),
        node_size=node_size,
        node_attr=(colormap=cmap, colorrange=(vmin, vmax)),
        edge_color=(:gray60, 0.5),
        edge_width=edge_width,
    )
    Colorbar(pos[1, 2]; colormap=cmap, limits=(vmin, vmax), height=Relative(0.75))
end

# Two barycenters in column 3, one per row
bar_specs = [
    (1, bar_unw,  rc_unw,  L"\nu_\lambda\ \ (\delta_g = 10^{-10})"),
    (2, bar_unw6, rc_unw6, L"\nu_\lambda\ \ (\delta_g = 10^{-6})"),
]

for (r, bar_data, rc_data, bar_title) in bar_specs
    gl = GridLayout(fig[r, 3])
    ax = Axis(gl[1, 1], title=bar_title, aspect=DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    vmin_b, vmax_b = extrema(bar_data)
    graphplot!(ax, g_plot;
        layout=geo_layout,
        node_color=bar_data,
        node_size=node_size,
        node_attr=(colormap=cmap, colorrange=(vmin_b, vmax_b)),
        edge_color=(:gray60, 0.5),
        edge_width=edge_width,
    )
    Colorbar(gl[1, 2]; colormap=cmap, limits=(vmin_b, vmax_b), height=Relative(0.6))
    Label(gl[2, 1:2], latexstring("\\lambda = $(round.(λ, sigdigits=2))"), tellwidth=false)
    Label(gl[3, 1:2], latexstring("\\hat{\\lambda} = $(round.(rc_data, sigdigits=3))"), tellwidth=false)
    rowgap!(gl, 1, 4)
    rowgap!(gl, 2, 2)
end

save("ma_house_barycenter.pdf", fig)
