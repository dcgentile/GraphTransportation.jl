include("./ExperimentUtils.jl")
using GraphTransportation
using CairoMakie, GraphMakie
using Graphs
using LinearAlgebra, SparseArrays, Statistics

Q, u = cube_markov_chain()
n_nodes = 8
n_measures = 4
A = Q .> 0
M = zeros(n_nodes, n_measures)
M[:,1] = random_geographic_concentration(A, center=1) ./ u
M[:,2] = random_geographic_concentration(A, center=3) ./ u
M[:,3] = random_geographic_concentration(A, center=6) ./ u
M[:,4] = random_geographic_concentration(A, center=8) ./ u

λ = rand(n_measures)
λ /= sum(λ)
bar = barycenter(M, λ, Q, h=0.5, tol=1e-10, geodesic_steps=10, maxiters=1024)
rc = analysis(bar, M, Q)
rel_err = norm(λ - rc) / norm(λ)

# Build SimpleGraph from adjacency matrix
g = SimpleGraph(n_nodes)
for i in 1:n_nodes, j in i+1:n_nodes
    A[i, j] != 0 && add_edge!(g, i, j)
end

# 2D perspective projection of cube nodes
cube_positions = [
    Point2f(0.0,  0.0),   # 1
    Point2f(1.0,  0.0),   # 2
    Point2f(1.0,  1.0),   # 3
    Point2f(0.0,  1.0),   # 4
    Point2f(0.35, 0.35),  # 5
    Point2f(1.35, 0.35),  # 6
    Point2f(1.35, 1.35),  # 7
    Point2f(0.35, 1.35),  # 8
]

cmap = :magma
fig = Figure(size=(1400, 700))

# Left half: reference measures in a 2×2 grid
panels = [
    (fig[1, 1], M[:, 1], "ν_1"),
    (fig[1, 2], M[:, 2], "ν_2"),
    (fig[2, 1], M[:, 3], "ν_3"),
    (fig[2, 2], M[:, 4], "ν_4"),
]

for (pos, nc, title) in panels
    ax = Axis(pos[1, 1], title=title, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    vmin, vmax = extrema(nc)
    graphplot!(ax, g;
        layout=(_) -> cube_positions,
        node_color=nc,
        node_size=15,
        node_attr=(colormap=cmap, colorrange=(vmin, vmax)),
        edge_color=:black,
    )
    Colorbar(pos[1, 2]; colormap=cmap, limits=(vmin, vmax))
end

# Right half: barycenter spanning both rows
bary_pos = fig[1:2, 3]
ax5 = Axis(bary_pos[1, 1],
    title="λ = $(round.(λ, sigdigits=4))\nRecovered Coordinates: $(round.(vec(rc), sigdigits=4))\n Relative Recovery Error: $(round(rel_err, sigdigits=4))",
    aspect=DataAspect())
hidedecorations!(ax5)
hidespines!(ax5)
vmin5, vmax5 = extrema(bar)
graphplot!(ax5, g;
    layout=(_) -> cube_positions,
    node_color=bar,
    node_size=15,
    node_attr=(colormap=cmap, colorrange=(vmin5, vmax5)),
    edge_color=:black,
)
Colorbar(bary_pos[1, 2]; colormap=cmap, limits=(vmin5, vmax5))

save("cube_barycenter.pdf", fig)
