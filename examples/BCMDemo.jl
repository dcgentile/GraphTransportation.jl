using GraphTransportation
using GraphMakie, Graphs, CairoMakie
using JLD2
using LaTeXStrings
include("./ExperimentUtils.jl")

Q, u, geo_cx, geo_cy = load_usa_mc()
A = Q .> 0
n_nodes = size(Q, 1)
n_measures = 3
M = zeros(n_nodes, n_measures)

weight = 10
M[:,1] = random_geographic_concentration(A, weight=weight, center=34) ./ u  # Arizona
M[:,2] = random_geographic_concentration(A, weight=weight, center=28) ./ u  # Illinois
M[:,3] = random_geographic_concentration(A, weight=weight, center=4)  ./ u  # Virginia

coords = [[1/3; 1/3; 1/3], [1/2; 1/2; 0], [0; 1/2; 1/2], [1/2; 0; 1/2],
          [2/3; 1/6; 1/6], [1/6; 2/3; 1/6], [1/6; 1/6; 2/3]]

h        = 0.1
maxiters = 2^16
tol      = 1e-8
geo_steps = 3
geo_tol  = 1e-10

if isfile("usa_bars.jld2")
    bars = load("usa_bars.jld2")["bars"]
else
    bars = [barycenter(M, coord, Q, h=h, maxiters=maxiters, tol=tol,
                       geodesic_steps=geo_steps, geodesic_tol=geo_tol) for coord in coords]
    @save "usa_bars.jld2" bars
end

# Recover barycentric coordinates and linear systems for each computed barycenter
if isfile("bcm_analysis.jld2")
    d  = load("bcm_analysis.jld2")
    rc = d["rc"]
    systems = d["systems"]
else
    results  = [analysis(bars[i], M, Q, N=geo_steps, compute_condition=true, return_system=true) for i in eachindex(bars)]
    rc       = [r[1][:,1] for r in results]
    systems  = [r[2]      for r in results]
    @save "bcm_analysis.jld2" rc systems
end

# SimpleGraph for graphplot!
g_plot = SimpleGraph(n_nodes)
for i in 1:n_nodes, j in i+1:n_nodes
    A[i, j] != 0 && add_edge!(g_plot, i, j)
end

# Geographic node positions from shapefile centroids
node_positions = [Point2f(geo_cx[i], geo_cy[i]) for i in 1:n_nodes]
geo_layout = (_) -> node_positions

# Convert densities (f = μ/u) to probability measures (μ = f .* u) for display.
# M[:,i] and bars[i] are densities; multiplying by u recovers the actual weights.
M_disp   = M .* u                          # n_nodes × n_measures
bars_disp = [b .* u for b in bars]         # each sums to 1

# Shared color range across all 7 panels
gmin, gmax = extrema(vcat(vec(M_disp), vcat(bars_disp...)))
cmap = :magma

# --- Figure setup ---
fig_w, fig_h = 1800, 1600
fig = Figure(size=(fig_w, fig_h), backgroundcolor=:white)

# Triangle vertex positions in normalized [0,1] coords (y=0 is bottom)
T1 = (0.50, 0.86)   # top:          measure 1
T2 = (0.13, 0.12)   # bottom-left:  measure 2
T3 = (0.87, 0.12)   # bottom-right: measure 3

# Convert barycentric weights to normalized figure coords
bary_to_norm(λ) = (λ[1]*T1[1] + λ[2]*T2[1] + λ[3]*T3[1],
                   λ[1]*T1[2] + λ[2]*T2[2] + λ[3]*T3[2])

# Panel half-dimensions in pixels
phw, phh = 175, 130
panel_rect(nx, ny) = Rect2f(nx*fig_w - phw, ny*fig_h - phh, 2phw, 2phh)

# Background axis: draw triangle outline
bg = Axis(fig, bbox=Rect2f(0, 0, fig_w, fig_h))
hidedecorations!(bg); hidespines!(bg)
limits!(bg, 0, 1, 0, 1)
lines!(bg, [T1[1], T2[1], T3[1], T1[1]], [T1[2], T2[2], T3[2], T1[2]],
       color=:gray50, linewidth=2)

# Helper: add a graph panel at normalized figure position (nx, ny)
function add_panel!(data, nx, ny, title, subtitle="")
    ax = Axis(fig, bbox=panel_rect(nx, ny), title=title, titlesize=16,
              subtitle=subtitle, subtitlesize=14)
    hidedecorations!(ax); hidespines!(ax)
    graphplot!(ax, g_plot;
        layout=geo_layout,
        node_color=collect(Float64, data),
        node_size=20,
        node_attr=(colormap=cmap, colorrange=(gmin, gmax)),
        edge_color=(:gray60, 0.4),
        edge_width=5,
    )
end

# Vertex panels
add_panel!(M_disp[:,1], T1..., L"\nu_1")
add_panel!(M_disp[:,2], T2..., L"\nu_2")
add_panel!(M_disp[:,3], T3..., L"\nu_3")

# Barycenter panels
for (i, coord) in enumerate(coords)
    nx, ny = bary_to_norm(coord)
    title    = latexstring("\\lambda=$(round.(coord; sigdigits=3))")
    subtitle = latexstring("\\hat{\\lambda}=$(round.(rc[i]; sigdigits=3))")
    add_panel!(bars_disp[i], nx, ny, title, subtitle)
end

# Shared colorbar on the right
Colorbar(fig, bbox=Rect2f(fig_w - 115, 300, 80, fig_h - 600);
         colormap=cmap, limits=(gmin, gmax), ticklabelsize=18)

save("bcm.pdf", fig)
