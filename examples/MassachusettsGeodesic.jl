include("./ExperimentUtils.jl")
using GraphTransportation
using LaTeXStrings
using CairoMakie, GraphMakie
using Graphs
using LinearAlgebra, SparseArrays, Statistics
using Shapefile, LibGEOS

# --- Markov chain ---
Q, sstate = ma_house_markov_chain()
println("Loaded the Massachusetts Graph")
n_nodes = size(Q, 1)
A = Q .> 0

# --- Two endpoint measures at geographically distant centers ---
weight = 10
ρ0 = random_geographic_concentration(A, weight=weight, center=5)   ./ sstate
ρ1 = random_geographic_concentration(A, weight=weight, center=145) ./ sstate

# --- Geodesic ---
N = 10
println("Computing the geodesic between the measures")
geovec = discrete_transport(Q, ρ0, ρ1, N=N, progress=true)
ρ = geovec.vector.ρ   # (N+1) × n_nodes

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

# --- Plot: 5 evenly spaced frames along the geodesic ---
frame_indices = [1, N÷4 + 1, N÷2 + 1, 3N÷4 + 1, N + 1]
frame_labels  = [L"t = 0", L"t = 1/4", L"t = 1/2", L"t = 3/4", L"t = 1"]

cmap = :magma
# Use a shared color scale across all frames so mass transport is visible
vmin = minimum(ρ)
vmax = maximum(ρ)

fig = Figure(size=(2000, 500))

for (col, (idx, label)) in enumerate(zip(frame_indices, frame_labels))
    pos = fig[1, col]
    ax = Axis(pos[1, 1], title=label, aspect=DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    graphplot!(ax, g_plot;
        layout=geo_layout,
        node_color=collect(ρ[idx, :]),
        node_size=5,
        node_attr=(colormap=cmap, colorrange=(vmin, vmax)),
        edge_color=(:gray60, 0.5),
        edge_width=0.5,
    )
end

Colorbar(fig[1, length(frame_indices) + 1]; colormap=cmap, limits=(vmin, vmax))

save("ma_house_geodesic.pdf", fig)
