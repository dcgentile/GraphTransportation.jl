using GraphMakie, Graphs, NetworkLayout, CairoMakie
using Shapefile, LibGEOS
using LinearAlgebra, SparseArrays, Statistics
using ProgressMeter 
using SparseArrays
using LinearAlgebra
using JLD2, DelimitedFiles
using GraphTransportation
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

