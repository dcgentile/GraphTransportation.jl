include("./ExperimentUtils.jl")
using GraphTransportation
using LaTeXStrings
using CairoMakie, GraphMakie
using Graphs
using LinearAlgebra, SparseArrays, Statistics

# --- Unweighted Markov chain ---
Q_unw, u_unw = hypercube_markov_chain()
n_nodes = size(Q_unw, 1)
A = Q_unw .> 0

# --- Weighted Markov chain (arbitrary symmetric integer weights on same topology) ---
W_raw = rand(1:10, n_nodes, n_nodes); W_raw = W_raw + W_raw'
W = Float64.(A .* W_raw)
d = vec(sum(W, dims=2))
Q_w = W ./ reshape(d, :, 1)
u_w = d / sum(d)

# --- Reference measures (shared; built relative to unweighted stationary dist) ---
n_measures = 4
M = zeros(n_nodes, n_measures)
M_w = zeros(n_nodes, n_measures)
weight=10
M[:,1] = random_geographic_concentration(A, weight=weight, center=1)  ./ u_unw
M[:,2] = random_geographic_concentration(A, weight=weight, center=6)  ./ u_unw
M[:,3] = random_geographic_concentration(A, weight=weight, center=11) ./ u_unw
M[:,4] = random_geographic_concentration(A, weight=weight, center=15) ./ u_unw

M_w[:,1] = random_geographic_concentration(A, weight=weight, center=1)  ./ u_w
M_w[:,2] = random_geographic_concentration(A, weight=weight, center=6)  ./ u_w
M_w[:,3] = random_geographic_concentration(A, weight=weight, center=11) ./ u_w
M_w[:,4] = random_geographic_concentration(A, weight=weight, center=15) ./ u_w

λ = ones(n_measures) / n_measures

# --- Barycenters ---
h = 0.1
steps=10
tol=1e-10
geo_tol=1e-10
bar_unw = barycenter(M, λ, Q_unw, h=h, tol=tol, geodesic_tol=geo_tol, geodesic_steps=steps, maxiters=512)
bar_w   = barycenter(M_w, λ, Q_w,   h=h, tol=tol, geodesic_tol=geo_tol, geodesic_steps=steps, maxiters=512)
rc_unw = analysis(bar_unw, M, Q_unw, N=steps)[:,1]
rc_w = analysis(bar_w, M_w, Q_w, N=steps)[:,1]

# --- SimpleGraph for plotting ---
g = SimpleGraph(n_nodes)
for i in 1:n_nodes, j in i+1:n_nodes
    A[i, j] != 0 && add_edge!(g, i, j)
end

# 2D Schlegel diagram: outer cube (nodes 1–8) + inner cube (nodes 9–16)
hypercube_positions = [
    Point2f(0.0, 0.0),   # 1  outer front face
    Point2f(2.0, 0.0),   # 2
    Point2f(2.0, 2.0),   # 3
    Point2f(0.0, 2.0),   # 4
    Point2f(0.5, 0.5),   # 5  outer back face (perspective offset)
    Point2f(2.5, 0.5),   # 6
    Point2f(2.5, 2.5),   # 7
    Point2f(0.5, 2.5),   # 8
    Point2f(0.9, 0.9),   # 9  inner front face
    Point2f(1.7, 0.9),   # 10
    Point2f(1.7, 1.7),   # 11
    Point2f(0.9, 1.7),   # 12
    Point2f(1.1, 1.1),   # 13 inner back face
    Point2f(1.9, 1.1),   # 14
    Point2f(1.9, 1.9),   # 15
    Point2f(1.1, 1.9),   # 16
]

cmap = :plasma
fig = Figure(size=(1600, 900))

# --- Top half: 4 reference measures ---
for (c, (nc, title)) in enumerate(zip(eachcol(M), [L"\nu_1", L"\nu_2", L"\nu_3", L"\nu_4"]))
    pos = fig[1, c]
    ax = Axis(pos[1, 1], title=title, aspect=DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    vmin, vmax = extrema(nc)
    graphplot!(ax, g;
        layout=(_) -> hypercube_positions,
        node_color=collect(nc),
        node_size=16,
        node_attr=(colormap=cmap, colorrange=(vmin, vmax)),
        edge_color=:black,
    )
    Colorbar(pos[1, 2]; colormap=cmap, limits=(vmin, vmax))
end

# --- Bottom left: unweighted barycenter ---
gl_unw = GridLayout(fig[2, 1:2])
ax_unw = Axis(gl_unw[1, 1], title=L"\nu_\lambda\ \ \text{(unweighted)}", aspect=DataAspect())
hidedecorations!(ax_unw); hidespines!(ax_unw)
vmin_u, vmax_u = extrema(bar_unw)
graphplot!(ax_unw, g;
    layout=(_) -> hypercube_positions,
    node_color=bar_unw,
    node_size=16,
    node_attr=(colormap=cmap, colorrange=(vmin_u, vmax_u)),
    edge_color=:black,
)
Colorbar(gl_unw[1, 2]; colormap=cmap, limits=(vmin_u, vmax_u))
Label(gl_unw[2, 1:2], latexstring("\\lambda = $(round.(λ, sigdigits=3))"), tellwidth=false)
Label(gl_unw[3, 1:2], latexstring("\\hat{\\lambda} = $(round.(rc_unw, sigdigits=3))"), tellwidth=false)
rowgap!(gl_unw, 1, 4)
rowgap!(gl_unw, 2, 2)

# --- Bottom right: weighted barycenter (edge labels show weights) ---
gl_w = GridLayout(fig[2, 3:4])
ax_w = Axis(gl_w[1, 1], title=L"\nu_\lambda\ \ \text{(weighted)}", aspect=DataAspect())
hidedecorations!(ax_w); hidespines!(ax_w)
vmin_w, vmax_w = extrema(bar_w)
graphplot!(ax_w, g;
    layout=(_) -> hypercube_positions,
    node_color=bar_w,
    node_size=16,
    node_attr=(colormap=cmap, colorrange=(vmin_w, vmax_w)),
    edge_color=:black,
    elabels=[string(round(W[src(e), dst(e)], sigdigits=3)) for e in edges(g)],
    elabels_textsize=8,
)
Colorbar(gl_w[1, 2]; colormap=cmap, limits=(vmin_w, vmax_w))
Label(gl_w[2, 1:2], latexstring("\\lambda = $(round.(λ, sigdigits=3))"), tellwidth=false)
Label(gl_w[3, 1:2], latexstring("\\hat{\\lambda} = $(round.(rc_w, sigdigits=3))"), tellwidth=false)
rowgap!(gl_w, 1, 4)
rowgap!(gl_w, 2, 2)

save("hypercube_barycenter.pdf", fig)
