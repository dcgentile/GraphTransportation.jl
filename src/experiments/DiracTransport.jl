using LinearAlgebra, SparseArrays
using GraphTransportation
include("../MarkovChains.jl")
include("../CommonGraphs.jl")

## Two Points
"""
    diracs_on_two_points(;N=128, ε=0.)

Description of the function.

#TODO
"""
function diracs_on_two_points(;N=128, ε=0.)
    Q = [0. 1.; 1. 0.]
    μ = [2.; 0]
    ν = [0.; 2]
    a = [-1; 1;]
    b = [1; -1;]
    γ = discrete_transport(Q, μ + ε * a, ν + ε * b, N=N)
    return (γ, sqrt(action(γ)))
end


## Triangle

"""
    diracs_on_triangle(;N = 128, ε = 0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_triangle(;N = 128, tol=1e-10, σ=0.5, τ=0.5)
    Q, sstate = triangle_markov_chain()
    V = size(Q,1)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[3] = 1/sstate[3]

    γ = discrete_transport(Q, μ, ν, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action(γ)))
end


function triangle_with_tail(; N=128, tol=1e-10, σ=0.5, τ=0.5)
    Q, sstate = triangle_with_tail_markov_chain()
    V = size(Q,1)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]

    γ = discrete_transport(Q, μ, ν, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action(γ)))
    
end


function diracs_on_prism(;N=100, tol=1e-10, σ=0.5, τ=0.5)
    Q, sstate = triangular_prism_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[2] = 1/sstate[2]

    γ = discrete_transport(Q, μ, ν, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action(γ)))
    
end

## Square
"""
    diracs_on_square(;N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_square(;N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5)
    Q, sstate = square_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[2] = 1/sstate[2]

    a = [-1; 1/3; 1/3; 1/3;]
    b = [1/3; -1; 1/3; 1/3;]

    γ = discrete_transport(Q, μ + ε * a, ν + ε * b, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action(γ)))
end

## T Graph
"""
    diracs_on_T(;N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_T(;N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5)
    Q, sstate = T_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]
    a = [-1; 1/3; 1/3; 1/3;]
    b = [1/3; -1; 1/3; 1/3;]
    γ = discrete_transport(Q, μ + ε * a, ν + ε * b, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action(γ)))
end
 
function diracs_on_double_T(;N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5)
    Q, sstate = double_T_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[3] = 1/sstate[3]
    γ = discrete_transport(Q, μ, ν, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action()))
end
## 9x9 Grid

"""
    diracs_on_grid(; N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function diracs_on_grid(; N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5)
    Q, sstate = grid_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[2] = 1/sstate[2]
    ν[6] = 1/sstate[6]

    γ = discrete_transport(Q, μ, ν, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action(γ)))

end


function diracs_on_cube(; N=128, ε=0., tol=1e-10, σ=0.5, τ=0.5)
    Q, sstate = cube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    #ν[2] = 1/sstate[2]
    ν[7] = 1/sstate[7]

    γ = discrete_transport(Q, μ, ν, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action(γ)))
end


function diracs_on_hypercube(; N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5)
    Q, sstate = hypercube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    ν[15] = 1/sstate[15]

    γ = discrete_transport(Q, μ, ν, N=N, tol=tol, σ=σ, τ=τ)
    return (γ, sqrt(action(γ)))
end
