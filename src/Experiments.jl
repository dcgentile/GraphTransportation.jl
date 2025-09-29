using SparseArrays
include("utils.jl")
include("tests/Inclusion.jl")

"""
    gromov_convergence(N, n)

Description of the function.

#TODO
"""
function gromov_convergence(N, n, verbose=false, tol=1e-6)
    Q = (1/2) * Tridiagonal(ones(N-1), zeros(N), ones(N-1))
    Q[1,2] = 1
    Q[N,N-1] = 1
    μ = zeros(N)
    ν = zeros(N)
    sstate = steady_state_from_adjacency(Q)
    μ[1] = 1 / sstate[1]
    ν[N] = 1 / sstate[N]
    γ,d = BBD(Q, μ, ν, N=n, verbose=verbose, tol=tol)
    return (γ, d)
end

"""
    plot_midpoint(N, n)

Description of the function.

#TODO
"""
function plot_midpoint(N, n)
    c, d = gromov_convergence(N, n)
    ρ = c.vector.ρ
    f = Figure()
    ax = Axis(f[1,1])
    lines!(ax, collect(1:N), ρ[n ÷ 2,:])
    current_figure()

end

## Two Points
"""
    diracs_on_two_points(;N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_two_points(;N=128, ε=0., verbose=false)
    Q = [0. 1.; 1. 0.]
    μ = [2.; 0]
    ν = [0.; 2]
    a = [-1; 1;]
    b = [1; -1;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N=N, verbose=verbose)
    return γ, d
end


## Triangle

"""
    diracs_on_triangle(;N = 128, ε = 0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_triangle(;N = 128, ε = 0., tol=1e-10, verbose=false)
    Q = [0. 0.5 0.5; 0.5 0. 0.5; 0.5 0.5 0.]
    μ = [3.; 0; 0.]
    ν = [0.; 3; 0.]
    a = [-1; 1/2; 1/2;]
    b = [1/2; -1; 1/2;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N=N, tol=tol, verbose=verbose)
    return (γ, d)
end


function triangle_with_tail(; N=128, tol=1e-10, verbose=false)
    Q = [0. 1/2 1/2 0.;
         1/3 0  1/3 1/3;
         1/2 1/2 0 0;
         0. 1. 0. 0.]
    
    π = steady_state_from_adjacency(Q)
    μ = zeros(4)
    ν = zeros(4)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]

    γ, d = BBD(Q, μ, ν, N=N, tol=tol, verbose=verbose)
    return (γ, d)
    
end


## Square
"""
    diracs_on_square(;N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_square(;N=128, ε=0., verbose=false, tol=1e-6, σ=0.5, τ=0.5)
    Q = [0. 0.5 0. 0.5;
         0.5 0. 0.5 0.;
         0. 0.5 0. 0.5;
         0.5 0. 0.5 0.]
    μ = [4.; 0; 0.; 0.]
    ν = [0.; 4; 0.; 0.]
    a = [-1; 1/3; 1/3; 1/3;]
    b = [1/3; -1; 1/3; 1/3;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
end

## T Graph
"""
    diracs_on_T(;N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_T(;N=128, ε=0., verbose=false, tol=1e-6, σ=0.5, τ=0.5)
    Q = [0. 1. 0. 0.;
         1/3 0. 1/3 1/3;
         0. 1. 0. 0.;
         0. 1. 0. 0.]
    μ = [3.; 0; 0.; 3.]
    ν = [0.; 0; 3.; 3.]
    a = [-1; 1/3; 1/3; 1/3;]
    b = [1/3; -1; 1/3; 1/3;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
end
## 9x9 Grid

"""
    diracs_on_grid(; N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_grid(; N=128, ε=0., tol=1e-6, verbose=false)
    Q = Array(Tridiagonal(ones(7), zeros(8), ones(7)))
    Q[1, 8] = 1
    Q[8, 1] = 1
    Q = hcat(Q, zeros(8))
    Q = vcat(Q, zeros(9)')
    Q[9,2] = Q[2,9] = Q[9,4] = Q[4,9] = Q[9,6] = Q[6,9] = Q[9,8] = Q[8,9] = 1

    for (idx, row) in enumerate(eachrow(Q))
        z = sum(row)
        Q[idx, :] /= z
    end

    π = steady_state_from_adjacency(Q)
    μ = zeros(9)
    ν = zeros(9)
    μ[2] = 1/π[2]
    ν[6] = 1/π[6]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol)
    return (γ, d)

end

function diracs_on_cube(; N=128, ε=0., tol=1e-6, verbose=false)

    edge_list = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 6), (6, 7), (7, 8), (8, 5)
    ]

    Q = zeros(8,8)

    for e in edge_list
        i, j = e
        Q[i,j] = 1
        Q[j,i] = 1
    end

    for (idx, row) in enumerate(eachrow(Q))
        z = sum(row)
        Q[idx, :] /= z
    end

    π = steady_state_from_adjacency(Q)
    μ = zeros(8)
    ν = zeros(8)
    μ[1] = 1/π[1]
    ν[7] = 1/π[7]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol)
    return (γ, d)
end

function diracs_on_hypercube(; N=128, ε=0., tol=1e-6, verbose=false)

    edge_list = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 6), (6, 7), (7, 8), (8, 5),

        (1,9), (2, 10), (3, 11), (4, 12),
        (5, 13), (6, 14), (7, 15), (8, 16),

        (9, 10), (10, 11), (11, 12), (12, 9),
        (9, 13), (10, 14), (11, 15), (12, 16),
        (13, 14), (14, 15), (15, 16), (16, 13)
    ]

    Q = zeros(16,16)

    for e in edge_list
        i, j = e
        Q[i,j] = 1
        Q[j,i] = 1
    end

    for (idx, row) in enumerate(eachrow(Q))
        z = sum(row)
        Q[idx, :] /= z
    end

    π = steady_state_from_adjacency(Q)
    μ = zeros(16)
    ν = zeros(16)
    μ[1] = 1/π[1]
    ν[15] = 1/π[15]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol)
    return (γ, d)
end
"""
    barycenter_on_grid(; N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function barycenter_on_grid_EGD(; N=128, ε=0., verbose=false)
    Q = Array(Tridiagonal(ones(7), zeros(8), ones(7)))
    Q[1, 8] = 1
    Q[8, 1] = 1
    Q = hcat(Q, zeros(8))
    Q = vcat(Q, zeros(9)')
    Q[9,2] = Q[2,9] = Q[9,4] = Q[4,9] = Q[9,6] = Q[6,9] = Q[9,8] = Q[8,9] = 1

    for (idx, row) in enumerate(eachrow(Q))
        z = sum(row)
        Q[idx, :] /= z
    end

    π = steady_state_from_adjacency(Q)
    μ = zeros(9)
    ν = zeros(9)
    ρ = zeros(9)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]
    ρ[5] = 1/π[5]
    M = hcat(μ, ν, ρ)'
    λ = [1/3; 1/3; 1/3;]

    J = variance_functional(Q, M, λ, N)
    F(x) = J(surface_point(x, π))
    x0 = ones(8)
    return optimize(F, x0, LBFGS(), Optim.Options(x_reltol=1e-6, iterations=128))
end

"""
    export_grid_vars()

Description of the function.

#TODO
"""
function export_grid_vars()

    Q = Array(Tridiagonal(ones(7), zeros(8), ones(7)))
    Q[1, 8] = 1
    Q[8, 1] = 1
    Q = hcat(Q, zeros(8))
    Q = vcat(Q, zeros(9)')
    Q[9,2] = Q[2,9] = Q[9,4] = Q[4,9] = Q[9,6] = Q[6,9] = Q[9,8] = Q[8,9] = 1

    for (idx, row) in enumerate(eachrow(Q))
        z = sum(row)
        Q[idx, :] /= z
    end

    π = steady_state_from_adjacency(Q)
    μ = zeros(9)
    ν = zeros(9)
    ρ = zeros(9)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]
    ρ[5] = 1/π[5]
    M = hcat(μ, ν, ρ)'

    return (Q, π, M)
end


"""
    visualize_grid_barycenter(Q, measures; layout_alg=Spring(), node_size=15, colormap=:viridis, edge_width=1.0, edge_color=:gray80, filename=nothing, fixed_positions=nothing, colorbar=true)

Description of the function.

#TODO
"""
function visualize_grid_barycenter(Q, measures;
                                 layout_alg=Spring(),
                                 node_size=15,
                                 colormap=:viridis,
                                 edge_width=1.0,
                                 edge_color=:gray80,
                                 filename=nothing,
                                 fixed_positions=nothing,
                                 colorbar=true,
                                 )
    M, V = size(measures)
    fig = Figure(size = (400*M, 400))
    g = SimpleDiGraph(Q)
    for i = 0:M-1
        v = measures[i+1,:]
        ax = Axis(fig[1, 2 * i + 1], aspect=DataAspect())
        hidedecorations!(ax)
        hidespines!(ax)

        # Compute layout or use fixed positions
        positions = isnothing(fixed_positions) ?
            NetworkLayout.layout(layout_alg, g) :
            fixed_positions

        # Normalize node values for coloring
        vmin, vmax = extrema(v)
        normalized_v = (v .- vmin) ./ (vmax - vmin + eps())

        # Plot the graph
        graphplot!(ax, g,
                   layout=positions,
                   node_size=node_size,
                   node_color=v,
                   edge_width=edge_width,
                   edge_color=edge_color,
                   node_attr=(; colormap=colormap, colorrange=(vmin, vmax)))

        # Add colorbar if requested
        if colorbar
            Colorbar(fig[1, 2*i], limits=(vmin, vmax), colormap=colormap, label="Node Weight")
        end
    end

    if !isnothing(filename)
        save(filename, fig)
    end

    return fig
	
end

"""
    visualize_weighted_graph(Q, v; layout_alg=Spring(), node_size=15, colormap=:viridis, edge_width=1.0, edge_color=:gray80, filename=nothing, fixed_positions=nothing, colorbar=true, figure_size=(800, 600))

Description of the function.

#TODO
"""
function visualize_weighted_graph(Q, v;
                                 layout_alg=Spring(),
                                 node_size=15,
                                 colormap=:viridis,
                                 edge_width=1.0,
                                 edge_color=:gray80,
                                 filename=nothing,
                                 fixed_positions=nothing,
                                 colorbar=true,
                                 figure_size=(800, 600))
    # Create graph from adjacency matrix
    g = SimpleDiGraph(Q)

    # Set up the figure
    fig = Figure(; size=figure_size)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    # Compute layout or use fixed positions
    positions = isnothing(fixed_positions) ?
                NetworkLayout.layout(layout_alg, g) :
                fixed_positions

    # Normalize node values for coloring
    vmin, vmax = extrema(v)
    normalized_v = (v .- vmin) ./ (vmax - vmin + eps())

    # Plot the graph
    graphplot!(ax, g,
               layout=positions,
               node_size=node_size,
               node_color=v,
               edge_width=edge_width,
               edge_color=edge_color,
               node_attr=(; colormap=colormap, colorrange=(vmin, vmax)))

    # Add colorbar if requested
    if colorbar
        Colorbar(fig[1, 2], limits=(vmin, vmax), colormap=colormap, label="Node Weight")
    end

    # Save to file if requested
    if !isnothing(filename)
        save(filename, fig)
    end

    return fig, positions
end


function inclusion_test(N)
    c, d = diracs_on_square(N)
    Script_K_pre_indicator(c)
end
