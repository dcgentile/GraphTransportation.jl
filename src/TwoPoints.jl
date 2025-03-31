include("EarthMover.jl")
using ProgressMeter

function run_experiment(σ=0.5, τ=0.5, maxiters=2^32, tol=1e-10)
    Q = [0. 1.; 1. 0.]
    μ = [2.; 0]
    ν = [0.; 2]

    for i=2:12
        N = 2^i
        γ, d = BBD(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol)
        println("Approximated distance for h = 2^$(-i): d = $(d)")
        #is_in_CE_weakly(γ.vector.ρ, γ.vector.m, γ.cache.Q, γ.cache.π)
        #is_in_CE_strongly(γ.vector.ρ, γ.vector.m, γ.cache.Q)
    end

end

function is_geodesic(N)
    Q = [0. 1.; 1. 0.]
    μ = [2.; 0]
    ν = [0.; 2]
    γ, d = BBD(Q, μ, ν, N)
    distances = [0.]

    @showprogress for t=1:N
        η, d0 = BBD(Q, μ, γ.vector.ρ[t,:], N)
        push!(distances, d0/d)
    end

    return distances


end
