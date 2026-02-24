using CairoMakie
using LinearAlgebra, SparseArrays, Statistics
using GraphTransportation
include("../CommonGraphs.jl")

function experiment(;grid_size=3,
                    h=0.5, maxiters=1000, tol=1e-9,
                    geo_steps=10, geo_tol=1e-9)

    Q, sstate = grid_markov_chain(grid_size)

    function random_measure()
	    m = rand(size(Q,1))
        m /= sum(m)
        return m ./ sstate
    end

    ρ1 = random_measure()
    ρ2 = random_measure()
    ρ3 = random_measure()
    M = cat(ρ1, ρ2, ρ3, dims=2)
    coords = rand(3)
    coords /= sum(coords)

    bary, norm_diffs, vars = barycenter(M, coords, Q,
                                        h=h, maxiters=maxiters, tol=tol,
                                        geodesic_tol=geo_tol, geodesic_steps=geo_steps)


    return norm_diffs

end

function plot_convergence(;n=3, kwargs...)
    results = [experiment(; kwargs...) for _ in 1:n]

    fig = Figure()
    ax = Axis(fig[1, 1],
              xlabel="Iterations",
              ylabel="Norm difference between successive iterates",
              yscale=log10)

    for (i, nd) in enumerate(results)
        lines!(ax, 1:length(nd), nd, label="Run $i")
    end

    #Legend(fig[1, 2], ax)
    return fig
end
