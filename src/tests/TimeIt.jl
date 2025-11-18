using BenchmarkTools
using LinearAlgebra
using SparseArrays
using BlockBandedMatrices
include("../utils.jl")
include("../galerkin/ContinuityEnforcer.jl")
include("../galerkin/ProximalEqualityIndicator.jl")
include("../galerkin/ProximalAvgIndicator.jl")
include("../galerkin/ProximalSignIndicator.jl")
include("../galerkin/ProximalAction.jl")
include("../galerkin/KProjection.jl")

T = 100
N = 100
σ = 4
Q = (1/N) * (ones(N,N) - I(N)) # Q is a complete graph on N vertices
A = form_ceh_system(Q, T)
M = form_avg_system(T)
μ = ν = zeros(N)
i, j = rand(1:N), rand(1:N)
μ[i] = N
ν[j] = N

function continuity_setup(T, N, σ)
    ρ = σ * randn(T + 1, N)
    m = σ * randn(T, N, N)
    return (ρ, m)
end

function pei_setup(T, N, σ)
    q = σ * randn(T, N)
    ρ_avg = σ * randn(T, N)
    return (q, ρ_avg)
end

function pai_setup(T, N, σ)
    ρ = σ * randn(T + 1, N)
    ρ_avg = σ * randn(T, N)
    return (ρ, ρ_avg)
end

function psi_setup(T, N, σ)
    q = σ * randn(T, N)
    ρ_plus = σ * randn(T, N, N)
    ρ_minus = σ * randn(T, N, N)
    return (q, ρ_plus, ρ_minus)
end

function action_setup(T, N, σ)
    θ = σ * randn(T, N, N)
    m = σ * randn(T, N, N)
	return (θ, m)
end

function kproj_setup(T, N, σ)
    θ = σ * randn(T, N, N)
    ρ_plus = σ * randn(T, N, N)
    ρ_minus = σ * randn(T, N, N)
    return (θ, ρ_plus, ρ_minus)
end


#println("Benchmarking prox_Astar!")
#display(@benchmark prox_Astar!(θ, m) setup=((θ, m) = action_setup($T, $N, $σ)))
#println("\nBenchmarking prox_IJpm_star!")
#display(@benchmark proximal_IJpm_star!(q, ρ_plus, ρ_minus, $Q) setup=((q, ρ_plus, ρ_minus) = psi_setup($T, $N, $σ)))
#println("\nBenchmarking prox_IJavg_star!")
#display(@benchmark prox_IJavg_star!(ρ, ρ_avg, $μ, $ν, $M) setup=((ρ, ρ_avg) = pai_setup($T, $N, $σ)))
#println("\nBenchmarking proj_CE!")
#display(@benchmark proj_CE!(ρ, m, $μ, $ν, $Q, $A) setup=((ρ, m) = continuity_setup($T, $N, $σ)))
println("\nBenchmarking proj_K!")
display(@benchmark project_K!(ρ_minus, ρ_plus, θ) setup=((ρ_minus, ρ_plus, θ) = kproj_setup($T, $N, $σ)))
#println("\nBenchmarking prox_IJeq!")
#display(@benchmark project_IJeq!(ρ_avg, q) setup=((ρ_avg, q) = pei_setup($T, $N, $σ)))
#
