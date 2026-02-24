using Shapefile, LibGEOS
using LinearAlgebra, SparseArrays, Statistics
using ProgressMeter 
using SparseArrays
using LinearAlgebra
using JLD2, DelimitedFiles
using GraphTransportation
include("CommonGraphs.jl")


function cube_analysis(n;N=32,tol=1e-9, σ=0.5, τ=0.5)
    Q, sstate = cube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    ν[7] = 1/sstate[7]
    M = hcat(μ, ν)

    coords = Dict()
    γ, d = BBD(Q, μ, ν, N=N, tol=tol, σ=σ, τ=τ)

    for i=2:n
        targ = γ.vector.ρ[end ÷ i + 1,:]
        coords[i] = analysis(targ, M, Q, N=N, tol=tol) 
    end

    return coords
        

end


function cube_synthesis(;N=100, tol=1e-10, maxiters=1000)
    Q, sstate = cube_markov_chain()
    V = size(Q,1)

    μ = zeros(V)
    ν = zeros(V)

    μ[1] = 1/sstate[1]
    ν[7] = 1/sstate[7]

    initialization = (1/6) * ones(8)
    initialization[1] = 3.5
    initialization[7] = 3.5

    M = hcat(μ, ν)
    
    return barycenter(M, [0.5; 0.5], Q,
                      initialization=initialization,
                      geodesic_steps=N,
                      geodesic_tol=tol,
                      maxiters=maxiters)
    
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
