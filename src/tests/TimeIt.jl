using BenchmarkTools
using LinearAlgebra
include("../ErbarVector.jl")
include("../galerkin/Chambolle.jl")

"""
this file benchmarks each component of the Chambolle Pock routine, both the in-place versions and the reallocating versions
"""

# constants, allocate some vectors
# N nodes
N = 50
# T steps in geodesic
T = 100
#set Markov Kernel and measures
Q = (1/N) * (ones(N,N) - I(N)) # Q is a complete graph on N vertices
μ = ν = zeros(N)
i, j = rand(1:N), rand(1:N)
μ[i] = N
ν[j] = N

M = form_avg_system(T)
A = form_ceh_system(Q,T)

function time_prox_Astar()
    # time prox_Astar!
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_Astar!")
    @btime prox_Astar!(a.vector.θ, b.vector.m)
    # time prox_Astar
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_Astar")
    @btime c = prox_Astar(a.vector.θ, b.vector.m);

end

function time_prox_IJpm_star()
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_IJpm_star!")
    @btime proximal_IJpm_star!(a.vector.q, a.vector.ρ_minus, a.vector.ρ_plus, Q)
    # time prox_IJpm_star
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_IJpm_star")
    @btime c = proximal_IJpm_star(a.vector.q, a.vector.ρ_minus, a.vector.ρ_plus, Q);

end

function time_prox_IJavg_star()
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_IJpm_star!")
    @btime prox_IJavg_star!(a.vector.ρ, a.vector.ρ_avg, μ, ν, M);
    # time prox_IJpm_star
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_IJpm_star")
    @btime c = prox_IJavg_star(a.vector.ρ, a.vector.ρ_avg, μ, ν, M);
end

function time_prox_Fstar()
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_Fstar!")
    @btime prox_Fstar!(a,b)
    # time prox_IJpm_star
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_Fstar")
    @btime c = prox_Fstar(0.5,a,b);
end

function time_proj_CE()
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking proj_CE!")
    @btime proj_CE!(a.vector.ρ, a.vector.m, μ, ν, Q, A)
    # time prox_IJpm_star
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking proj_CE")
    _ = @btime proj_CE(a.vector.ρ, a.vector.m, μ, ν, Q, A);
end

function time_proj_K()
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking proj_K!")
    @btime project_K!(a.vector.ρ_minus, a.vector.ρ_plus, a.vector.θ)
    # time prox_IJpm_star
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking proj_K")
    @btime c = project_K(a.vector.ρ_minus, a.vector.ρ_plus, a.vector.θ);
end

function time_proj_IJeq()
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_IJeq!")
    @btime project_IJeq!(a.vector.ρ_avg, a.vector.q)
    # time prox_IJpm_star
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_IJpm_star")
    @btime c = project_IJeq(a.vector.ρ_avg, a.vector.q);
end

function time_prox_G()
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_G!")
    @btime prox_G!(a,b)
    # time prox_IJpm_star
    a = ErbarBundle(Q, μ, ν, T)
    b = ErbarBundle(Q, μ, ν, T)
    println("Benchmarking prox_G")
    @btime c = prox_G(0.5,a,b);
end

function time_pipeline()
    time_prox_Astar()
    time_prox_IJpm_star()
    time_prox_IJavg_star()
    time_prox_Fstar()
    time_proj_CE()
    time_proj_K()
    time_proj_IJeq()
    time_proxG()
end
