using GraphMakie, Graphs, NetworkLayout, CairoMakie
using Shapefile, LibGEOS
using LinearAlgebra, SparseArrays, Statistics
using ProgressMeter 
using SparseArrays
using LinearAlgebra
using JLD2, DelimitedFiles
using GraphTransportation
include("../CommonGraphs.jl")

function wametx_barycenter(;N_steps=16, tol=1e-8)
    Qusa, sstate = load_usa_mc()
    
    μ1 = zeros(49)
    μ2 = zeros(49)
    μ3 = zeros(49)
    μ1[38] = 1/sstate[38]
    μ2[12] = 1/sstate[12]
    μ3[15] = 1/sstate[15]
    M = stack((μ1, μ2, μ3));
    
    coords = ones(3) / 3

    if isfile("wamext_synth_outs.jld2")
        bary = load("wametx_synth_outs.jld2")["bary"]
    else
        bary, ndiffs, variances = barycenter(M, coords, Qusa,
                                             maxiters=1000, geodesic_tol=tol, geodesic_steps=N_steps)
        @save "wametx_synth_outs.jld2" bary ndiffs variances N_steps tol

    end

    recovered_coords = analysis(bary, M, Qusa, N=100, tol=tol)

    @save "wametx_analysis_outs.jld2" recovered_coords tol
    
    return (bary, ndiffs, variances, recovered_coords)
    
end

function sampta_barycenter(;N_steps=16, tol=1e-8)
    Qusa, sstate = load_usa_mc()
    M = sstate.^-1 .* readdlm("sampta_references.txt")
    coords = [0.5; 0.3; 0.2]

    if isfile("sampta_synth_outs.jld2")
        bary = load("sampta_synth_outs.jld2")["bary"]
    else
        bary, ndiffs, variances = barycenter(M, coords, Qusa,
                                             maxiters=1000, geodesic_tol=tol, geodesic_steps=N_steps,
                                             h=0.1)
        @save "sampta_synth_outs.jld2" bary ndiffs variances N_steps tol
    end


    recovered_coords = analysis(bary, M, Qusa, N=100, tol=tol)

    @save "sampta_analysis_outs.jld2" recovered_coords tol
    
    return (bary, ndiffs, variances, recovered_coords)
end
                                         
function sampta_barycenter2(;N_steps=16, tol=1e-9)
    Qusa, sstate = load_usa_mc()
    M = sstate.^-1 .* readdlm("sampta2_references.txt")
    coords = [0.3; 0.2; 0.1; 0.1; 0.2; 0.1]

    if isfile("sampta2_synth_outs.jld2")
        bary = load("sampta2_synth_outs.jld2")["bary"]
    else
        bary, ndiffs, variances = barycenter(M, coords, Qusa,
                                             maxiters=1000, geodesic_tol=tol, geodesic_steps=N_steps)
        @save "sampta2_synth_outs.jld2" bary ndiffs variances N_steps tol
    end


    recovered_coords = analysis(bary, M, Qusa, N=100, tol=tol)

    @save "sampta2_analysis_outs.jld2" recovered_coords tol
    
    return (bary, ndiffs, variances, recovered_coords)
end

function load_usa_mc()
    function kernel_from_adjacency(adj)
        N, _ = size(adj)
        A = sparse(adj)
        nedges = nnz(A)
        degree_vector = A * ones(N)
        sstate = degree_vector / nedges
        Q = zeros(size(A))
        for i = 1:N, j = 1:N
            if A[i,j] !=0 
                Q[i,j] = 1/ (sstate[i]*nedges)
            end
        end
        return Q, sstate
    end

    shapes = Shapefile.Handle("./data/states.shp").shapes
    n = length(shapes)
    
    # Get adjacency matrix
    adj = [touches(shapes[i], shapes[j]) || intersects(shapes[i], shapes[j]) 
           for i in 1:n, j in 1:n]
    
    # Get centroids from shape points directly
    cx = Float64[]
    cy = Float64[]
    for shape in shapes
        points = shape.points
        push!(cx, mean(p.x for p in points))
        push!(cy, mean(p.y for p in points))
    end
    
    Qusa, sstate = kernel_from_adjacency(adj - I(49)) 
    return Qusa, sstate
end
