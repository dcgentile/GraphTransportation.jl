using JLD2
using LinearAlgebra
using CairoMakie
using GraphTransportation

function experiment(n_experiments;grid_size=4,
                    h=0.5, maxiters=1000, tol=1e-10,
                    geo_steps=10, geo_tol=1e-10)

    Q, sstate = grid_markov_chain(grid_size)

    function random_measure(n)
	    m = rand(n)
        m /= sum(m)
        return m ./ sstate
    end

    function experiment_instance()
        n_nodes = size(Q,1)
        ρ1 = random_measure(n_nodes)
        ρ2 = random_measure(n_nodes)
        ρ3 = random_measure(n_nodes)
        M = cat(ρ1, ρ2, ρ3, dims=2)
        coords = rand(size(M,2))
        coords /= sum(coords)

        bary, norm_diffs, _ = barycenter(M, coords, Q,
                        h=h, maxiters=maxiters, tol=tol,
                        geodesic_tol=geo_tol, geodesic_steps=geo_steps,
                        return_stats=true)
	    n_iters = size(norm_diffs,1)
        rcs = analysis(bary, M, Q)
        return n_iters, norm(coords - rcs) / norm(coords)
    end

    iter_counts = zeros(n_experiments)
    rel_errs = zeros(n_experiments)
    for i=1:n_experiments
        if i % (n_experiments ÷ 10) == 0
            @save "SynthAnalyProg_$(i).jld2" iter_counts rel_errs
        end
        iter_counts[i], rel_errs[i] = experiment_instance()
    end
    return iter_counts, rel_errs
end

function plot_results(results)
    f = Figure()
    ax = Axis(f[1,1], xlabel="Relative Error", ylabel="Observations")
    hist!(ax, results)
    f
end

function plot_counts(results)
    f = Figure()
    ax = Axis(f[1,1], xlabel="Descent Iterations to Convergence", ylabel="Observations")
    hist!(ax, results)
    f
end

function run_and_plot(N, grid_size; h=0.25)
	iter_counts, rel_errs = experiment(N, grid_size=grid_size, h=h)
    f_errs = plot_results(rel_errs)
    f_counts = plot_counts(iter_counts)
    save("analysis_recovery.pdf", f_errs)
    save("convergence_iterations.pdf", f_counts)
    f_errs
end
