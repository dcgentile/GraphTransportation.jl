using GraphTransportation
using LinearAlgebra, SparseArrays, Statistics
using CairoMakie, GraphMakie, Graphs
using LaTeXStrings
using Printf
include("./ExperimentUtils.jl")

"""
Module 2 validation (spec.txt §2.3.2/§2.3.3): compare `barycenter_socp` against the
existing intrinsic-descent `barycenter` on the hypercube (weighted & unweighted) and
the 7x7 (49-node) grid, reporting:
  - ‖ν*_descent - ν*_SOCP‖_π and wall-clock ratio
  - the objective certificate J_SOCP ≤ J at the descent solution (global optimality
    of the convex SOCP solve)
and rendering a side-by-side comparison figure per graph.
"""

norm_pi(v, π) = sqrt(sum(π .* v .^ 2))

# Warm up JuMP/Clarabel's JIT compilation on a throwaway problem before any timed run:
# the first call to geodesic_socp/barycenter_socp in a session pays for compiling the
# @variable/@constraint/RotatedSecondOrderCone machinery, which can dwarf the actual
# solve time on small problems (measured ~8s compile vs ~0.2s actual solve on the
# hypercube case below) and would otherwise make whichever case runs first look
# artificially slow relative to the others.
let
    G_warm = MarkovGraph([0.0 1.0; 1.0 0.0], [0.5, 0.5])
    geodesic_socp(G_warm, [1.5, 0.5], [0.5, 1.5]; N=2)
    barycenter_socp(G_warm, [[1.5, 0.5], [0.5, 1.5]], [0.5, 0.5]; N=2)
end

function objective_at(G::MarkovGraph, refs, λ, ν; N)
    sum(λ[i] * geodesic_socp(G, refs[i], ν; N=N).W2 for i in eachindex(refs))
end

function run_case(name, Q, π, M, λ; N=10, descent_kwargs=NamedTuple())
    G = MarkovGraph(Q, π)
    refs = [M[:, i] for i in 1:size(M, 2)]

    t_descent = @elapsed bar_descent = barycenter(M, λ, Q;
        h=0.1, geodesic_steps=N, verbose=false, descent_kwargs...)
    t_socp = @elapsed (ν_socp, J_socp, geos_socp) = barycenter_socp(G, refs, λ; N=N)

    discrepancy = norm_pi(bar_descent .- ν_socp, π)
    J_descent = objective_at(G, refs, λ, bar_descent; N=N)
    certificate_ok = J_socp <= J_descent + 1e-6 * max(abs(J_descent), 1.0)

    @printf("== %s ==\n", name)
    @printf("  wall-clock: descent=%.3fs  socp=%.3fs  ratio(descent/socp)=%.2fx\n",
            t_descent, t_socp, t_descent / t_socp)
    @printf("  ‖ν_descent - ν_socp‖_π = %.6g\n", discrepancy)
    @printf("  J_socp = %.6g   J_descent(evaluated via SOCP geodesics) = %.6g   certificate (J_socp <= J_descent): %s\n",
            J_socp, J_descent, certificate_ok)

    return (; G, refs, λ, bar_descent, ν_socp, J_socp, J_descent, discrepancy,
              t_descent, t_socp, certificate_ok)
end

function plot_comparison(path, title_str, g, positions, refs, bar_descent, ν_socp, λ; node_size=28, cmap=:plasma, fontsize=24)
    p = length(refs)
    fig = Figure(size=(300 * (p + 1), 640))

    for (c, ref) in enumerate(refs)
        pos = fig[1, c]
        ax = Axis(pos[1, 1], title=latexstring("\\nu_$c"), titlesize=fontsize, aspect=DataAspect())
        hidedecorations!(ax); hidespines!(ax)
        vmin, vmax = extrema(ref)
        graphplot!(ax, g; layout=(_) -> positions, node_color=ref, node_size=node_size,
                   node_attr=(colormap=cmap, colorrange=(vmin, vmax)), edge_color=:black)
        Colorbar(pos[1, 2]; colormap=cmap, limits=(vmin, vmax))
    end

    gl_descent = GridLayout(fig[2, 1:cld(p, 2)])
    ax_d = Axis(gl_descent[1, 1], title=L"\nu_\lambda\ \text{(intrinsic descent)}", titlesize=fontsize, aspect=DataAspect())
    hidedecorations!(ax_d); hidespines!(ax_d)
    vmin_d, vmax_d = extrema(bar_descent)
    graphplot!(ax_d, g; layout=(_) -> positions, node_color=bar_descent, node_size=node_size,
               node_attr=(colormap=cmap, colorrange=(vmin_d, vmax_d)), edge_color=:black)
    Colorbar(gl_descent[1, 2]; colormap=cmap, limits=(vmin_d, vmax_d))

    gl_socp = GridLayout(fig[2, cld(p, 2)+1:p])
    ax_s = Axis(gl_socp[1, 1], title=L"\nu_\lambda\ \text{(SOCP)}", titlesize=fontsize, aspect=DataAspect())
    hidedecorations!(ax_s); hidespines!(ax_s)
    vmin_s, vmax_s = extrema(ν_socp)
    graphplot!(ax_s, g; layout=(_) -> positions, node_color=ν_socp, node_size=node_size,
               node_attr=(colormap=cmap, colorrange=(vmin_s, vmax_s)), edge_color=:black)
    Colorbar(gl_socp[1, 2]; colormap=cmap, limits=(vmin_s, vmax_s))

    Label(fig[0, 1:p], title_str, fontsize=fontsize+4)
    save(path, fig)
    return path
end

# ── Hypercube: unweighted & weighted ─────────────────────────────────────────────
Q_unw, π_unw = hypercube_markov_chain()
n_nodes = size(Q_unw, 1)
A = Q_unw .> 0

W_raw = rand(1:10, n_nodes, n_nodes); W_raw = W_raw + W_raw'
W = Float64.(A .* W_raw)
d = vec(sum(W, dims=2))
Q_w = W ./ reshape(d, :, 1)
π_w = d / sum(d)

M_unw = zeros(n_nodes, 4)
M_w = zeros(n_nodes, 4)
for (c, center) in enumerate((1, 6, 11, 15))
    M_unw[:, c] = random_geographic_concentration(A, weight=10, center=center) ./ π_unw
    M_w[:, c]   = random_geographic_concentration(A, weight=10, center=center) ./ π_w
end
λ_hc = fill(1/4, 4)

hypercube_positions = [
    Point2f(0.0, 0.0), Point2f(2.0, 0.0), Point2f(2.0, 2.0), Point2f(0.0, 2.0),
    Point2f(0.5, 0.5), Point2f(2.5, 0.5), Point2f(2.5, 2.5), Point2f(0.5, 2.5),
    Point2f(0.9, 0.9), Point2f(1.7, 0.9), Point2f(1.7, 1.7), Point2f(0.9, 1.7),
    Point2f(1.1, 1.1), Point2f(1.9, 1.1), Point2f(1.9, 1.9), Point2f(1.1, 1.9),
]
g_hc = SimpleGraph(n_nodes)
for i in 1:n_nodes, j in i+1:n_nodes
    A[i, j] != 0 && add_edge!(g_hc, i, j)
end

res_unw = run_case("hypercube (unweighted)", Q_unw, π_unw, M_unw, λ_hc; N=10)
res_w   = run_case("hypercube (weighted)",   Q_w,   π_w,   M_w,   λ_hc; N=10)

plot_comparison("hypercube_unweighted_socp_vs_descent.pdf", "Hypercube (unweighted)",
                 g_hc, hypercube_positions, [M_unw[:,i] for i in 1:4], res_unw.bar_descent, res_unw.ν_socp, λ_hc)
plot_comparison("hypercube_weighted_socp_vs_descent.pdf", "Hypercube (weighted)",
                 g_hc, hypercube_positions, [M_w[:,i] for i in 1:4], res_w.bar_descent, res_w.ν_socp, λ_hc)

# ── 7x7 grid (49 nodes) ──────────────────────────────────────────────────────────
Q_grid, π_grid = grid_markov_chain(7)
n_grid = size(Q_grid, 1)
A_grid = Q_grid .> 0
M_grid = zeros(n_grid, 3)
for (c, center) in enumerate((1, 25, 49))
    M_grid[:, c] = random_geographic_concentration(A_grid, weight=10, center=center) ./ π_grid
end
λ_grid = [0.5, 0.3, 0.2]

grid_positions = [Point2f(((i-1) % 7), -((i-1) ÷ 7)) for i in 1:n_grid]
g_grid = SimpleGraph(n_grid)
for i in 1:n_grid, j in i+1:n_grid
    A_grid[i, j] != 0 && add_edge!(g_grid, i, j)
end

res_grid = run_case("7x7 grid", Q_grid, π_grid, M_grid, λ_grid; N=10)

plot_comparison("grid_socp_vs_descent.pdf", "7×7 Grid",
                 g_grid, grid_positions, [M_grid[:,i] for i in 1:3], res_grid.bar_descent, res_grid.ν_socp, λ_grid;
                 node_size=18, fontsize=20)

println("\nAll figures saved to $(pwd())")
