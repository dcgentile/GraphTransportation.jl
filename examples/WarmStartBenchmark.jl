using GraphTransportation
using GraphMakie, Graphs, CairoMakie
using JLD2, Printf
using LinearAlgebra, SparseArrays
include("./ExperimentUtils.jl")

# ---------------------------------------------------------------------------
# Shared barycenter parameters
# ---------------------------------------------------------------------------
coord    = [1/3; 1/3; 1/3]
h        = 0.1
maxiters = 256
tol      = 1e-8
geo_steps = 8
geo_tol  = 1e-6

function run_pair(Q, u, M, label)
    println("\n=== $label ===")

    t_cold = @elapsed ν_cold = barycenter(M, coord, Q;
                                    h=h, maxiters=maxiters, tol=tol,
                                    geodesic_steps=geo_steps, geodesic_tol=geo_tol,
                                    geodesic_warmstart=false, verbose=true)

    t_warm = @elapsed ν_warm = barycenter(M, coord, Q;
                                    h=h, maxiters=maxiters, tol=tol,
                                    geodesic_steps=geo_steps, geodesic_tol=geo_tol,
                                    geodesic_warmstart=true,  verbose=true)

    d = norm((ν_cold - ν_warm) .* sqrt.(u))

    @printf("  cold:  %.3f s\n", t_cold)
    @printf("  warm:  %.3f s\n", t_warm)
    @printf("  ratio: %.2fx %s\n", t_cold/t_warm,
            t_warm < t_cold ? "(warm faster)" : "(cold faster)")
    @printf("  u-norm distance between outputs: %.2e\n", d)

    return ν_cold, ν_warm, t_cold, t_warm, d
end

# ---------------------------------------------------------------------------
# Experiment 1: 4×4 grid
# ---------------------------------------------------------------------------
grid_size = 4
Q_grid, u_grid = grid_markov_chain(grid_size)
A_grid = Q_grid .> 0
n_grid = size(Q_grid, 1)

weight  = 10
centers = [1, 4, 13]
M_grid  = zeros(n_grid, length(centers))
for i in eachindex(centers)
    M_grid[:,i] = random_geographic_concentration(A_grid, weight=weight, center=centers[i]) ./ u_grid
end

ν_cold_grid, ν_warm_grid, t_cold_grid, t_warm_grid, d_grid =
    run_pair(Q_grid, u_grid, M_grid, "4×4 Grid")

# ---------------------------------------------------------------------------
# Experiment 2: USA graph
# ---------------------------------------------------------------------------
Q_usa, u_usa, geo_cx, geo_cy = load_usa_mc()
A_usa = Q_usa .> 0
n_usa = size(Q_usa, 1)

M_usa = zeros(n_usa, 3)
M_usa[:,1] = random_geographic_concentration(A_usa, weight=weight, center=34) ./ u_usa  # Arizona
M_usa[:,2] = random_geographic_concentration(A_usa, weight=weight, center=28) ./ u_usa  # Illinois
M_usa[:,3] = random_geographic_concentration(A_usa, weight=weight, center=4)  ./ u_usa  # Virginia

ν_cold_usa, ν_warm_usa, t_cold_usa, t_warm_usa, d_usa =
    run_pair(Q_usa, u_usa, M_usa, "USA Graph (49 nodes)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
println("\n=== Summary ===")
@printf("  %-22s  cold: %6.3f s  warm: %6.3f s  ratio: %.2fx  ‖Δν‖_u = %.2e\n",
        "4×4 Grid",  t_cold_grid, t_warm_grid, t_cold_grid/t_warm_grid, d_grid)
@printf("  %-22s  cold: %6.3f s  warm: %6.3f s  ratio: %.2fx  ‖Δν‖_u = %.2e\n",
        "USA Graph", t_cold_usa,  t_warm_usa,  t_cold_usa/t_warm_usa,  d_usa)

# ---------------------------------------------------------------------------
# Figure: 2 rows × 2 cols per graph (cold | warm), one block per graph
# ---------------------------------------------------------------------------
cmap = :magma

function make_figure(Q, u, A, M, ν_cold, ν_warm, positions, title)
    g = SimpleGraph(size(Q, 1))
    for i in 1:size(Q,1), j in i+1:size(Q,1)
        A[i,j] != 0 && add_edge!(g, i, j)
    end
    layout = (_) -> positions

    M_disp     = M .* u
    cold_disp  = ν_cold .* u
    warm_disp  = ν_warm .* u
    gmin, gmax = extrema(vcat(vec(M_disp), cold_disp, warm_disp))

    fig = Figure(size=(900, 560), figure_padding=10)
    Label(fig[0, 1:3], title, fontsize=14, tellwidth=false)

    function panel!(row, col, data, ptitle)
        ax = Axis(fig[row, col], title=ptitle, titlesize=11)
        hidedecorations!(ax); hidespines!(ax)
        graphplot!(ax, g;
            layout=layout,
            node_color=collect(Float64, data),
            node_size=15,
            node_attr=(colormap=cmap, colorrange=(gmin, gmax)),
            edge_color=(:gray60, 0.4),
            edge_width=2,
        )
    end

    # Row 1: input measures
    for i in 1:size(M, 2)
        panel!(1, i, M_disp[:,i], "ν$(i)")
    end

    # Row 2: cold | warm barycenter
    panel!(2, 1, cold_disp, "barycenter (cold)")
    panel!(2, 2, warm_disp, "barycenter (warm)")

    Colorbar(fig[1:2, size(M,2)+1]; colormap=cmap, limits=(gmin, gmax))
    return fig
end

# Grid figure
grid_positions = [Point2f((k-1) % grid_size, (k-1) ÷ grid_size) for k in 1:n_grid]
fig_grid = make_figure(Q_grid, u_grid, A_grid, M_grid,
                       ν_cold_grid, ν_warm_grid, grid_positions,
                       "4×4 Grid — cold vs warm-started barycenter")
save("warmstart_grid.pdf", fig_grid)
println("Saved warmstart_grid.pdf")

# USA figure
usa_positions = [Point2f(geo_cx[i], geo_cy[i]) for i in 1:n_usa]
fig_usa = make_figure(Q_usa, u_usa, A_usa, M_usa,
                      ν_cold_usa, ν_warm_usa, usa_positions,
                      "USA Graph — cold vs warm-started barycenter")
save("warmstart_usa.pdf", fig_usa)
println("Saved warmstart_usa.pdf")
