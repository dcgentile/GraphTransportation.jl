using SparseArrays
include("utils.jl")
include("tests/Inclusion.jl")

"""
    gromov_convergence(N, n)

Description of the function.

#TODO
"""
function gromov_convergence(N, n, verbose=false, tol=1e-6)
    edge_list = [(i, i+1) for i=1:N-1]
    Q, π = markov_chain_from_edge_list(edge_list)
    μ = zeros(N)
    ν = zeros(N)
    μ[1] = 1 / π[1]
    ν[N] = 1 / π[N]
    γ,d = BBD(Q, μ, ν, N=n, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
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
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N=N, verbose=verbose, σ=σ, τ=τ)
    return γ, d
end


## Triangle

"""
    diracs_on_triangle(;N = 128, ε = 0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_triangle(;N = 128, ε = 0., tol=1e-10, σ=0.5, τ=0.5, verbose=false)
    edge_list = [(1,2), (2, 3), (3, 1)]

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)
    
    π = steady_state_from_adjacency(Q)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]

    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N=N, tol=tol, verbose=verbose, σ=σ, τ=τ)
    return (γ, d)
end


function triangle_with_tail(; N=128, tol=1e-10, verbose=false, σ=0.5, τ=0.5)

    edge_list = [(1,2), (2, 3), (3, 1), (3, 4)]

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)
    
    π = steady_state_from_adjacency(Q)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]

    γ, d = BBD(Q, μ, ν, N=N, tol=tol, verbose=verbose, σ=σ, τ=τ)
    return (γ, d)
    
end


function diracs_on_prism(;N=100, tol=1e-10, σ=0.5, τ=0.5, verbose=false)

    edge_list = [
        (1,2), (2, 3), (3, 1),
        (1,4), (2, 5), (3, 6),
        (4,5), (5, 6), (6, 4)
    ]

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[2] = 1/π[2]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
    
end

## Square
"""
    diracs_on_square(;N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_square(;N=128, ε=0., verbose=false, tol=1e-6, σ=0.5, τ=0.5)
    edge_list = [(1,2), (2, 3), (3, 4), (4, 1)]

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[2] = 1/π[2]

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
    edge_list = [(1,2), (2, 3), (2, 4)]

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]
    a = [-1; 1/3; 1/3; 1/3;]
    b = [1/3; -1; 1/3; 1/3;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
end
 
function diracs_on_double_T(;N=128, ε=0., verbose=false, tol=1e-6, σ=0.5, τ=0.5)
    edge_list = [
        (1,2), (2, 3), (2, 4),
        (1,5), (2, 6), (3, 7), (4, 8),
        (5,6), (6, 7), (6, 8),
    ]

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]
    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
end
## 9x9 Grid

"""
    diracs_on_grid(; N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_grid(; N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5, verbose=false)

    edge_list = [
        (1, 2), (2, 3), (3, 4),
        (3, 4), (4, 5), (5, 6),
        (6, 7), (7, 8), (8, 1),
        (9, 2), (9, 4), (9, 6), (9, 8)
    ]

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[2] = 1/π[2]
    ν[6] = 1/π[6]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)

end

function diracs_on_cube(; N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5, verbose=false)

    edge_list = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 6), (6, 7), (7, 8), (8, 5)
    ]

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/π[1]
    ν[2] = 1/π[2]
    #ν[7] = 1/π[7]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
end

function diracs_on_hypercube(; N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5, verbose=false)

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

    Q, π = markov_chain_from_edge_list(edge_list)
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)


    μ[1] = 1/π[1]
    ν[15] = 1/π[15]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
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

function inclusion_test(N)
    c, d = diracs_on_square(N)
    Script_K_pre_indicator(c)
end
