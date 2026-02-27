using GraphMakie, Graphs, NetworkLayout, CairoMakie
using LinearAlgebra, SparseArrays, Statistics
using ProgressMeter 
using JLD2 
using GraphTransportation
include("../CommonGraphs.jl")

function generate_barycenters(grid_size, n_steps;
                              h=0.5, maxiters=1000, tol=1e-9,
                              geo_steps=16, geo_tol=1e-9)

    Q, sstate = grid_markov_chain(grid_size)

    function random_measure()
	    m = rand(size(Q,1))
        m /= sum(m)
        return m ./ sstate
    end

    ρ0 = random_measure()
    ρ1 = random_measure()
    M = cat(ρ0, ρ1, dims=2)

    geovec = discrete_transport(Q, ρ0, ρ1, N=100)
    coords = [[1 - i/10; i/10] for i=1:9]
    barys = [barycenter(M, coord, Q,
                        maxiters=maxiters, tol=tol,
                        h=h, geodesic_tol=geo_tol, geodesic_steps=geo_steps,) for coord in coords]
    norm_diffs = [norm(geovec.vector.ρ[i*10 + 1,:] - barys[i]) for i=1:9]

    @save "grid_bars.jld2" grid_size n_steps maxiters tol M geovec barys 

    return (geovec.vector.ρ, barys, norm_diffs)
	
end
