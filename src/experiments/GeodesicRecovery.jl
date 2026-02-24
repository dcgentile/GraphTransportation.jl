using CairoMakie
using LinearAlgebra, SparseArrays, Statistics
using JLD2 
using GraphTransportation
include("../CommonGraphs.jl")

function experiment(;
                    h=0.25, maxiters=200, tol=1e-10,
                    geo_steps=10, geo_tol=1e-8)

    Q, sstate = triangle_markov_chain()

    function random_measure()
	    m = rand(size(Q,1))
        m /= sum(m)
        return m ./ sstate
    end

    ρ0 = random_measure()
    ρ1 = random_measure()
    M = cat(ρ0, ρ1, dims=2)
    N = 100
    geovec, _ = BBD(Q, ρ0, ρ1, N=N)
    coords = [[1 - i/N; i/N] for i=1:N-1]
    barys = [barycenter(M, coord, Q, maxiters=maxiters, tol=tol, h=h, geodesic_tol=geo_tol, geodesic_steps=geo_steps)[1] for coord in coords]
    norm_diffs = [norm(geovec.vector.ρ[i + 1,:] - barys[i]) / norm(geovec.vector.ρ[i+ 1,:]) for i=1:N-1]

	return norm_diffs
end

function experiment_randomized(;
                    h=0.5, maxiters=1000, tol=1e-10,
                    geo_steps=10, geo_tol=1e-8)

    Q, sstate = triangle_markov_chain()
    N = 100

    function random_measure()
	    m = rand(size(Q,1))
        m /= sum(m)
        return m ./ sstate
    end

    function random_experiment()
        ρ0 = random_measure()
        ρ1 = random_measure()
        M = cat(ρ0, ρ1, dims=2)
        geovec, _ = BBD(Q, ρ0, ρ1, N=N)
        t = rand(1:N)
        coords = [1 - t/N; t/N]
        bary, _, _ = barycenter(M, coords, Q, maxiters=maxiters, tol=tol, h=h, geodesic_tol=geo_tol, geodesic_steps=geo_steps)

        return norm(geovec.vector.ρ[t + 1,:] - bary) / norm(geovec.vector.ρ[t + 1,:])
    end

    norm_diffs = [random_experiment() for _ in 1:1000]

	return norm_diffs
end

function plot_results(results)
    f = Figure()
    ax = Axis(f[1,1], xlabel="Relative Error", ylabel="Observations")
    hist!(ax, results)
    f
end
