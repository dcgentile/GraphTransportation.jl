# TutorialFigures.jl
#
# Static figures for docs/src/tutorial.md, on the same 3×3 grid and the same densities
# the tutorial's executed blocks use. Two figures:
#   tutorial_geodesic.png    the SOCP geodesic ρA → ρB (N=20) at t = 0, 1/4, 1/2, 3/4, 1
#   tutorial_barycenter.png  the three references and their (0.5, 0.3, 0.2)-barycenter
# Panels show masses ρ ∘ π on a shared color scale. Run from src/experiments/ with
# --project=.; writes the PNGs into docs/src/assets/.

using GraphTransportation
using GraphMakie, Graphs, CairoMakie
using LinearAlgebra, Printf

Q, π = grid_markov_chain(3); n = 9
G = MarkovGraph(Q, π)
normalize_π(v) = v ./ dot(v, π)
ρA = normalize_π([4.0, 2, 1, 2, 1, 0.5, 1, 0.5, 0.25])
ρB = normalize_π(reverse(ρA))
ρC = normalize_π([0.5, 4.0, 0.5, 1, 2, 1, 0.5, 1, 0.5])
refs = [ρA, ρB, ρC]; λ = [0.5, 0.3, 0.2]

g_plot = SimpleGraph(n)
for (x, y) in G.E; add_edge!(g_plot, x, y); end
positions = [Point2f((i - 1) % 3, -((i - 1) ÷ 3)) for i in 1:n]     # node i at (column, −row)
cmap = :plasma
out = joinpath(@__DIR__, "..", "..", "docs", "src", "assets")

function panel!(fig, pos, mass, title, vmax)
    ax = Axis(fig[pos...], title=title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    graphplot!(ax, g_plot; layout=(_) -> positions, node_color=mass, node_size=42,
               node_attr=(colormap=cmap, colorrange=(0, vmax)), edge_color=(:gray55, 0.8), edge_width=2.5)
    limits!(ax, -0.4, 2.4, -2.4, 0.4)
end

# ---- figure 1: geodesic snapshots ----
N = 20
geo = geodesic(G, ρA, ρB; N=N)
ts = (0.0, 0.25, 0.5, 0.75, 1.0)
masses = [geo.ρ[:, round(Int, t * N) + 1] .* π for t in ts]
vmax = maximum(maximum.(masses))
fig = Figure(size=(1400, 340), backgroundcolor=:white)
for (k, (t, m)) in enumerate(zip(ts, masses))
    panel!(fig, (1, k), m, @sprintf("t = %.2f", t), vmax)
end
Colorbar(fig[1, 6], colormap=cmap, limits=(0, vmax), label="mass", labelsize=16)
Label(fig[0, :], @sprintf("Discrete transport geodesic from ρA to ρB on the 3×3 grid (SOCP, N=%d), W² = %.4f", N, geo.W2),
      fontsize=18, tellwidth=false)
save(joinpath(out, "tutorial_geodesic.png"), fig; px_per_unit=1)

# ---- figure 2: references and barycenter ----
ν, J, _ = barycenter(G, refs, λ; N=10)
λ̂ = vec(analysis(G, ν, refs; N=10))
masses = [r .* π for r in refs]; push!(masses, ν .* π)
vmax = maximum(maximum.(masses))
fig = Figure(size=(1320, 380), backgroundcolor=:white)
for k in 1:3
    panel!(fig, (1, k), masses[k], @sprintf("reference %d  (λ = %.1f)", k, λ[k]), vmax)
end
panel!(fig, (1, 4), masses[4], @sprintf("barycenter, J = %.4f\nλ̂ = (%.3f, %.3f, %.3f)", J, λ̂...), vmax)
Colorbar(fig[1, 5], colormap=cmap, limits=(0, vmax), label="mass", labelsize=16)
save(joinpath(out, "tutorial_barycenter.png"), fig; px_per_unit=1)
println("Saved tutorial_geodesic.png and tutorial_barycenter.png to docs/src/assets/")
