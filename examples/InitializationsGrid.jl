include("./ExperimentUtils.jl")
using GraphTransportation
using LaTeXStrings
using LinearAlgebra
using CairoMakie, GraphMakie, Graphs
using JLD2

grid_n = 5
Q, u = grid_markov_chain(grid_n)
n_nodes = size(Q, 1)
n_measures = 5
sqrt_u = sqrt.(u)
A = Q .> 0

# Corners and center of the grid
centers = [1, grid_n, grid_n*(grid_n-1)+1, grid_n^2, (grid_n^2+1)÷2]
center_labels = [L"\text{Top-left}", L"\text{Top-right}", L"\text{Bottom-left}", L"\text{Bottom-right}", L"\text{Center}"]
M = hcat([random_geographic_concentration(A, weight=8, center=c) ./ u for c in centers]...)

tol = 1e-10
geodesic_tol = 1e-6
geodesic_steps = 16
h = 0.25
maxiters = 2048

println("Parameters:")
println("  Geodesic steps per geodesic : $geodesic_steps")
println("  Geodesic convergence tol    : $geodesic_tol")
println("  Descent step size           : $h")
println("  Descent convergence tol     : $tol")

# Build SimpleGraph from adjacency matrix for visualization
g = SimpleGraph(n_nodes)
for i in 1:n_nodes, j in i+1:n_nodes
    A[i, j] != 0 && add_edge!(g, i, j)
end

# Grid layout: node k is at row ceil(k/grid_n), col ((k-1) % grid_n) + 1
grid_positions = [Point2f((k-1) % grid_n, -((k-1) ÷ grid_n)) for k in 1:n_nodes]

cache_file = "initializations_grid.jld2"

if isfile(cache_file)
    println("Loading cached results from $cache_file ...")
    @load cache_file bars vars λ max_bar_discrepancy min_W_dist
else
    λ = rand(n_measures)
    λ /= sum(λ)

    bars = Any[]
    vars = Any[]

    for i in 1:n_measures
        bar, nds, variance = barycenter(
            M, λ, Q,
            h=h, geodesic_steps=geodesic_steps,
            geodesic_tol=geodesic_tol, tol=tol, maxiters=maxiters,
            return_stats=true, initialization_index=i,
            verbose=true, geodesic_warmstart=true
        )
        push!(vars, variance[findlast(!iszero, variance)])
        append!(bars, bar)
    end

    bar_vecs_all = [bars[(i-1)*n_nodes+1 : i*n_nodes] for i in 1:n_measures]
    max_bar_discrepancy = maximum(
        norm((bar_vecs_all[i] - bar_vecs_all[j]) .* sqrt_u)
        for i in 1:n_measures for j in i+1:n_measures
    )

    min_W_dist = minimum(
        transport_cost(Q, M[:,i], M[:,j], N=geodesic_steps, tol=geodesic_tol)
        for i in 1:n_measures for j in i+1:n_measures
    )

    @save cache_file bars vars λ max_bar_discrepancy min_W_dist
    println("Saved results to $cache_file")
end

function plot_initializations_grid(M, bars, vars, g, positions, n_nodes;
                                   cmap=:plasma, titles=nothing,
                                   max_bar_discrepancy=nothing, min_W_dist=nothing,
                                   geodesic_steps=nothing, geodesic_tol=nothing,
                                   h=nothing, tol=nothing)
    n_measures = size(M, 2)
    bar_vecs = [bars[(i-1)*n_nodes+1 : i*n_nodes] for i in 1:n_measures]

    mmin, mmax = extrema(vec(M))
    bmin, bmax = extrema(bars)

    fig = Figure(size=(300 * n_measures, 950))

    for i in 1:n_measures
        # Top row: reference measure
        label = isnothing(titles) ? latexstring("\\nu_{$i}") : titles[i]
        ax_top = Axis(fig[1, i], title=label)
        hidedecorations!(ax_top)
        hidespines!(ax_top)
        graphplot!(ax_top, g;
            layout=(_) -> positions,
            node_color=M[:, i],
            node_size=16,
            node_attr=(colormap=cmap, colorrange=(mmin, mmax)),
            edge_color=:black,
        )

        # Bottom row: barycenter
        ax_bot = Axis(fig[3, i])
        hidedecorations!(ax_bot)
        hidespines!(ax_bot)
        graphplot!(ax_bot, g;
            layout=(_) -> positions,
            node_color=bar_vecs[i],
            node_size=16,
            node_attr=(colormap=cmap, colorrange=(bmin, bmax)),
            edge_color=:black,
        )

        # Variance label below each barycenter
        Label(fig[4, i], latexstring("\\mathcal{J} = $(round(vars[i], digits=5))"), tellwidth=false)
    end

    # Middle row: diagram showing the barycenter computation
    ax_mid = Axis(fig[2, 1:n_measures])
    rowsize!(fig.layout, 2, Relative(0.28))
    hidespines!(ax_mid)
    hidedecorations!(ax_mid)
    xlims!(ax_mid, 0, 1)
    ylims!(ax_mid, 0, 1)

    cx      = 0.5
    box_hw  = 0.20
    box_top = 0.82
    box_bot = 0.18
    gap     = 0.02

    # Arrow from above into the box
    arrows!(ax_mid, [cx], [1.0], [0.0], [box_top - 1.0 + gap];
        arrowsize=10, color=:black)

    # Center box
    poly!(ax_mid,
        [Point2f(cx-box_hw, box_bot), Point2f(cx+box_hw, box_bot),
         Point2f(cx+box_hw, box_top), Point2f(cx-box_hw, box_top)];
        color=:white, strokecolor=:black, strokewidth=1.5)

    y_mid = (box_top + box_bot) / 2

    # Build parameter string for the box
    param_lines = String[]
    !isnothing(geodesic_steps) && push!(param_lines, "geodesic steps = $geodesic_steps")
    !isnothing(geodesic_tol)   && push!(param_lines, "geodesic tol = $geodesic_tol")
    !isnothing(h)              && push!(param_lines, "step size = $h")
    !isnothing(tol)            && push!(param_lines, "descent tol = $tol")
    param_str = join(param_lines, ",  ")

    has_disc = !isnothing(max_bar_discrepancy)
    y_main = has_disc ? y_mid + 0.16 : y_mid + 0.08
    text!(ax_mid, cx, y_main - 0.13;
        text=L"Compute barycenter of $M$ with weights $\lambda$ and initialization $\nu_i$",
        align=(:center, :center), fontsize=20)

    if has_disc && !isnothing(min_W_dist)
        text!(ax_mid, cx, y_main - 0.26;
            text=latexstring(
                "\\max_{i\\neq j}\\Vert\\hat{\\nu}_i-\\hat{\\nu}_j\\Vert_{L^2(u)}\\approx" *
                "$(round(max_bar_discrepancy, digits=4))"
            ),
            align=(:center, :center), fontsize=20)
    end

    # Arrow from below out of the box
    arrows!(ax_mid, [cx], [box_bot - gap], [0.0], [-box_bot + gap];
        arrowsize=10, color=:black)

    Colorbar(fig[1, n_measures+1]; colormap=cmap, limits=(mmin, mmax))
    Colorbar(fig[3, n_measures+1]; colormap=cmap, limits=(bmin, bmax))

    return fig
end

fig = plot_initializations_grid(M, bars, vars, g, grid_positions, n_nodes,
    titles=center_labels,
    max_bar_discrepancy=max_bar_discrepancy, min_W_dist=min_W_dist,
    geodesic_steps=geodesic_steps, geodesic_tol=geodesic_tol,
    h=h, tol=tol)
save("initializations_grid.pdf", fig)

function plot_barycenter_heatmap(bars, vars, n_nodes, sqrt_u; cmap=:viridis)
    n = length(vars)
    bar_vecs = [bars[(i-1)*n_nodes+1 : i*n_nodes] for i in 1:n]
    D = [norm((bar_vecs[i] - bar_vecs[j]) .* sqrt_u) for i in 1:n, j in 1:n]
    println("Maximum pairwise barycenter distance: $(round(maximum(D), digits=10))")

    fig = Figure()
    ax = Axis(fig[1, 1],
        title="Pairwise barycenter distance (grid)",
        xticks=(1:n, string.(1:n)),
        yticks=(1:n, string.(1:n)),
    )
    hm = heatmap!(ax, D; colormap=cmap)
    Colorbar(fig[1, 2], hm)
    return fig
end

fig_heatmap = plot_barycenter_heatmap(bars, vars, n_nodes, sqrt_u)
save("initializations_grid_heatmap.pdf", fig_heatmap)
