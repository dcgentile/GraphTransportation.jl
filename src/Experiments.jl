using SparseArrays
using LinearAlgebra
using JLD2
using Convex, SCS
include("tests/Inclusion.jl")
include("CommonGraphs.jl")

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
function diracs_on_triangle(;N = 128, tol=1e-10, σ=0.5, τ=0.5, verbose=false)
    Q, sstate = triangle_markov_chain()
    V = size(Q,1)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[3] = 1/sstate[3]

    γ, d = BBD(Q, μ, ν, N=N, tol=tol, verbose=verbose, σ=σ, τ=τ)
    return (γ, d)
end


function triangle_with_tail(; N=128, tol=1e-10, verbose=false, σ=0.5, τ=0.5)
    Q, sstate = triangle_with_tail_markov_chain()
    V = size(Q,1)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/π[1]
    ν[3] = 1/π[3]

    γ, d = BBD(Q, μ, ν, N=N, tol=tol, verbose=verbose, σ=σ, τ=τ)
    return (γ, d)
    
end


function diracs_on_prism(;N=100, tol=1e-10, σ=0.5, τ=0.5, verbose=false)
    Q, sstate = triangular_prism_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[2] = 1/sstate[2]

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
    Q, sstate = square_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[2] = 1/sstate[2]

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
    Q, sstate = T_markov_chain()
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
    Q, sstate = double_T_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[3] = 1/sstate[3]
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
    Q, sstate = grid_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[2] = 1/sstate[2]
    ν[6] = 1/sstate[6]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)

end



function diracs_on_cube(; N=128, ε=0., tol=1e-10, σ=0.5, τ=0.5, verbose=false)
    Q, sstate = cube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    #ν[2] = 1/sstate[2]
    ν[7] = 1/sstate[7]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
end

function diracs_on_hypercube(; N=128, ε=0., tol=1e-6, σ=0.5, τ=0.5, verbose=false)
    Q, sstate = hypercube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    ν[15] = 1/sstate[15]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    return (γ, d)
end
"""
    barycenter_on_grid(; N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function barycenter_on_grid_EGD(; N=128, ε=0., verbose=false)
    Q, sstate = grid_markov_chain()

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

function cube_analysis(n;N=32, verbose=false, tol=1e-9, σ=0.5, τ=0.5)
    Q, sstate = cube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    ν[7] = 1/sstate[7]

    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol, σ=σ, τ=τ)
    targ = γ.vector.ρ[end ÷ n + 1,:]
     
    #M = hcat(μ, ν)
    M = Diagonal(sstate.^-1)

    analysis(targ, M, Q)
    
end

function cube_synthesis(;N=32, verbose=false, tol=1e-9, σ=0.5, τ=0.5, maxiters=100)
    Q, sstate = cube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    ν[7] = 1/sstate[7]

    M = hcat(μ, ν)
    
    bar = barycenter(M, [0.5; 0.5], Q, geodesic_steps=N, geodesic_tol=tol, maxiters=maxiters)
    
end


function cube_trouble(;N=32, tol=1e-9, verbose=false)
    Q, sstate = cube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    ν[7] = 1/sstate[7]

    M = hcat(μ, ν)
    γ, d = BBD(Q, μ, ν, N=N, verbose=verbose, tol=tol)

    targ = γ.vector.ρ[end ÷ 2 + 1,:]

    γ1, _ = BBD(Q, targ, μ, N=N, tol=tol)
    γ2, _ = BBD(Q, targ, ν, N=N, tol=tol)

    return γ, γ1, γ2

    
end


function objective_scaling(Q, μ, ν, l, u, filename)
    M = hcat(μ, ν)
    p = size(M, 2)

    true_coords = [0.5; 0.5]

    coord_errs = []
    objs = []
    normalized_objs = []
    gradient_norms = []
    gram_mat_norms = []

    for k in l:u
        N = 2^k
        target = BBD(Q, μ, ν, N=N)[1].vector.ρ[end ÷ 2 + 1,:]

        tangent_vectors = [BBD(Q, target, M[:,i], N=N, tol=1e-10)[1].vector.m[1,:,:] for i=1:p]
        append!(gradient_norms, norm(0.5*(tangent_vectors[1] + 0.5*tangent_vectors[2])))
        g = metric_tensor(target)
        
        # form the matrix A for the QP
        A = zeros(p,p)
        for i=1:p, j=i:p
            A[i,j] = A[j,i] = sum(tangent_vectors[i] .* tangent_vectors[j] .* g)
        end
        
        
        # solve the QP
        n = size(A, 1)
        x = Variable(n)
        problem = minimize(quadform(x, A))
        # Simplex constraints
        problem.constraints = vcat(problem.constraints, [x >= 0])
        problem.constraints = vcat(problem.constraints, [sum(x) == 1])
        
        solve!(problem, SCS.Optimizer)
        
        append!(coord_errs, norm(x.value - true_coords) / norm(true_coords))  # optimal solution
        append!(objs, x.value' * A * x.value)
        append!(normalized_objs, (x.value' * A * x.value) / norm(A) )
        append!(gram_mat_norms, norm(A))
        
    end

    results = (coord_errs, objs, normalized_objs, gradient_norms, gram_mat_norms)
    @save filename results

end

function cube_objective_scaling()
    Q, sstate = cube_markov_chain()
    V = size(Q,1)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[7] = 1/sstate[7]
    objective_scaling(Q, μ, ν, 3, 8, "cube_scaling_results.jld2")
end

function triangle_objective_scaling()
    Q, sstate = triangle_markov_chain()
    V = size(Q,1)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[3] = 1/sstate[3]
    objective_scaling(Q, μ, ν, 3, 8, "triangle_scaling_results.jld2")
end
