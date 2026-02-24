using CairoMakie
using GraphTransportation
include("../CommonGraphs.jl")

function experiment(n_experiments;grid_size=3,
                    h=0.5, maxiters=1000, tol=1e-9,
                    geo_steps=10, geo_tol=1e-9)

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

        bary, norm_diffs, vars = barycenter(M, coords, Q,
                                            h=h, maxiters=maxiters, tol=tol,
                                            geodesic_tol=geo_tol, geodesic_steps=geo_steps)
	    n_iters = size(norm_diffs,1)
        rcs = analysis(bary, M, Q)
        return n_iters, norm(coords - rcs) / norm(coords)
    end

    iter_counts = zeros(n_experiments)
    rel_errs = zeros(n_experiments)
    for i=1:n_experiments
        iter_counts[i], rel_errs[i] = experiment_instance()
    end
    return iter_counts, rel_errs
end
