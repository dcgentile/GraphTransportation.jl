using LinearAlgebra, SparseArrays, Statistics
using JLD2, DelimitedFiles
using GraphTransportation
include("../CommonGraphs.jl")

function grid_barycenter(;N_steps=16, tol=1e-8)
    Qusa, sstate = grid_markov_chain(7)
    M = sstate.^-1 .* readdlm("references_grid.txt")
    coords = [0.5; 0.3; 0.2]

    if isfile("grid_synth_outs.jld2")
        bary = load("grid_synth_outs.jld2")["bary"]
    else
        bary, ndiffs, variances = barycenter(M, coords, Qusa,
                                             maxiters=1000, geodesic_tol=tol, geodesic_steps=N_steps,
                                             h=0.1)
        @save "grid_synth_outs.jld2" bary ndiffs variances N_steps tol
    end


    recovered_coords = analysis(bary, M, Qusa, N=100, tol=tol)

    @save "grid_analysis_outs.jld2" recovered_coords tol
    
    return (bary .* sstate , ndiffs, variances, recovered_coords)
end
                                         
