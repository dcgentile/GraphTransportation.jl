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
    barycenter_on_grid(; N=128, ε=0., verbose=false)

Description of the function.

#TODO
"""
function barycenter_on_grid_EGD(; N=128, ε=0., verbose=false)
    Q, sstate = grid_markov_chain()

    μ = zeros(9)
    ν = zeros(9)
    ρ = zeros(9)

    μ[1] = 1/π[1]
    ν[3] = 1/π[3]
    ρ[5] = 1/π[5]

    M = hcat(μ, ν, ρ)'
    λ = [1/3; 1/3; 1/3;]

    J = variance_functional(Q, M, λ, N)
    F(x) = J(surface_point(x, π))
    x0 = ones(8)
    return optimize(F, x0, LBFGS(), Optim.Options(x_reltol=1e-6, iterations=128))
end


function midpoint_distance()
    Q, sstate = cube_markov_chain()
    V = size(Q,1)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[7] = 1/sstate[7]

    dists_μ = []
    dists_ν = []

    for k in 3:8
        c, d = BBD(Q, μ, ν, N=2^k)
        midpoint = c.vector.ρ[end ÷ 2 + 1, :]
        c_μ, d_μ = BBD(Q, midpoint, μ, N=2^k)
        c_ν, d_ν = BBD(Q, midpoint, ν, N=2^k)
        append!(dists_μ, d_μ)
        append!(dists_ν, d_ν)
    end

    return (dists_μ, dists_ν)

end

function cube_midpoint_vectors()
    Q, sstate = cube_markov_chain()
    V = size(Q,1)
    μ = zeros(V)
    ν = zeros(V)
    μ[1] = 1/sstate[1]
    ν[7] = 1/sstate[7]

    norms = []
    
    for k in 3:11
        c, d = BBD(Q, μ, ν, N=2^k)
        mp = c.vector.ρ[end ÷ 2 + 1, :]
        cb, _ = BBD(Q, mp, μ, N=2^k)
        cf, _ = BBD(Q, mp, ν, N=2^k)
        tv = 0.5 * (cb.vector.m[1,:,:] + cf.vector.m[1,:,:])
        stat = norm(tv)
        append!(norms, stat)
    end
    
    @save "cube_tv_norms.jld2" norms
    
end



function linear_barycenter_experiment(;n_steps=100, tol=1e-10)
    Qusa, sstate = load_usa_mc()
    M = sstate.^-1 .* readdlm("sampta_references.txt")
    coords = [0.5; 0.3; 0.2]

    if isfile("sampta_synth_outs.jld2")
        bary = load("sampta_synth_outs.jld2")["bary"]
    else
        bary, ndiffs, variances = barycenter(M, coords, Qusa,
                                             maxiters=1000, geodesic_tol=tol, geodesic_steps=n_steps)
        @save "sampta_synth_outs.jld2" bary ndiffs variances n_steps tol
    end

    ν = ones(size(Qusa,1))
    tangent_vector = zeros(size(Qusa))
    p = size(M, 2)
    variance = 0
    for i=1:p
        gamma, dist =  BBD(Qusa, ν, M[:, i], N = n_steps, tol=tol)
        tangent_vector = tangent_vector + coords[i] * (gamma.vector.m[1,:,:])
        variance = variance + 0.5 * coords[i] * dist^2
    end

    linearized_barycenter = ν .- graph_divergence(Qusa, metric_tensor(ν) .* tangent_vector)
    
    recovered_coords = analysis(bary, M, Qusa, N=100, tol=tol)
    linearized_recovered_coords = analysis(linearized_barycenter, M, Qusa, N=100, tol=tol)

    @save "linearized_sampta_analysis_outs.jld2" recovered_coords tol
    
    return (bary, recovered_coords, linearized_barycenter, linearized_recovered_coords)
	
end


function hyperparameter_error_experiment(;
                                         grid_size=3,
                                         n_measures=3,
                                         synth_steps=[4,8,16],
                                         synth_tols=[1e-8,1e-9,1e-10],
                                         analysis_steps=100,
                                         analysis_tol=1e-10,
                                         fresh_run=false)
    Q, sstate = grid_markov_chain(grid_size)
    n_nodes = grid_size^2

    function sample_simplex(p)
        x = -log.(rand(p))
        x / sum(x)
    end

    function get_details()
        detail_filename = "err_exp_geos/details.jld2"
        if !isfile(detail_filename) || fresh_run
            weights = sample_simplex(n_measures)
            M = zeros(n_nodes, n_measures)
            for i=1:n_measures
                M[:,i] = sample_simplex(n_nodes)
            end
            M = sstate.^-1 .* M
            @save detail_filename weights M
        else
            weights = load(detail_filename)["weights"]
            M = load(detail_filename)["M"]
        end
	    return (weights, M)
    end

    weights, M = get_details()

    function get_bary(n, steps, tol)
        filename = "err_exp_geos/$(n)_$(steps)_$(tol).jld2"
        if !isfile(filename) || fresh_run
            bary, ndiffs, variances = barycenter(M, weights, Q,
                                                 maxiters=1000, geodesic_tol=tol, geodesic_steps=steps)
            @save filename bary
        else
            bary = load(filename)["bary"]
        end
        return bary
    end

    function get_error(measure)
        rcs = analysis(measure, M, Q, N=analysis_steps, tol=analysis_tol)
        return norm(rcs - weights) / norm(weights)
    end

    error_matrix = zeros(size(synth_steps,1), size(synth_tols,1))
    Threads.@threads for idx in LinearIndices(error_matrix)
        i, j = Tuple(CartesianIndices(error_matrix)[idx])
        error_matrix[i,j] = get_error(get_bary(n_nodes, synth_steps[i], synth_tols[j]))
    end

    return error_matrix
end

function two_point_barycenter_gd_experiment(;n_samples=1000, N=100, tol=1e-10)
    function generate_sample(;n_measures=2,
                             synth_steps=16, synth_tol=1e-10,
                             analysis_steps=100, analysis_tol=1e-10,
                             maxiters=1000, bary_tol=1e-10)
        
        Q = [0 1; 1 0]
        sstate = [1/2; 1/2]
        
        function random_measure()
            t = rand()
            s = 1-t
            return [t;s]
        end
        
        μ0 = sstate.^-1 .* random_measure()
        μ1 = sstate.^-1 .* random_measure()
        M = hcat(μ0, μ1)
        coord_index = rand(1:synth_steps-1)
        coords = [1 - coord_index / synth_steps; coord_index/synth_steps]
        
        g, _ = BBD(Q, μ0, μ1, N=synth_steps, tol=synth_tol)
        
        μt = g.vector.ρ[coord_index + 1,:]
        rcs = analysis(μt, M, Q, N=analysis_steps, tol=analysis_tol)
        re = norm(coords - rcs) / norm(coords)
        
        μt_hat, _, _ = barycenter(M, coords, Q,
                                  geodesic_tol=synth_tol, geodesic_steps=synth_steps,
                                  maxiters=maxiters, tol=bary_tol)
        rcs_hat = analysis(μt_hat, M, Q, N=analysis_steps, tol=analysis_tol)
        re_hat = norm(coords - rcs_hat) / norm(coords)
        
        #return (M, coords, μt, μt_hat, rcs, rcs_hat, re, re_hat)
        return ([re; re_hat], μt, μt_hat)
    end

    errors = zeros(n_samples, 2)
    geodesic_outputs = zeros(n_samples,2)
    barycentric_outputs = zeros(n_samples,2)
    Threads.@threads for i in 1:n_samples
        outs = generate_sample(synth_steps=N, synth_tol=tol)
        errors[i,:] = outs[1]
        geodesic_outputs[i,:] = outs[2]
        barycentric_outputs[i,:] = outs[3]
    end

    @save "2pt_outs.jld2" errors geodesic_outputs barycentric_outputs

    return errors, geodesic_outputs, barycentric_outputs
	
end


function barycenter_on_triangle(;
                             synth_steps=128, synth_tol=1e-8,
                             maxiters=1000, bary_tol=1e-10, h=1.0)
    Q, sstate = triangle_markov_chain()
    V = size(Q,1)
    μ1 = zeros(V)
    μ2 = zeros(V)
    μ3 = zeros(V)
    μ1[1] = 1/sstate[1]
    μ2[2] = 1/sstate[2]
    μ3[3] = 1/sstate[3]
    M = hcat(μ1, μ2, μ3)

    weights = [1/3; 1/6; 1/2;]
    bar, _, _ = barycenter(M, weights, Q,
                              geodesic_tol=synth_tol, geodesic_steps=synth_steps,
                              maxiters=maxiters, tol=bary_tol, h=h)

    return bar

end
