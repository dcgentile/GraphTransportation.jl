using CairoMakie
using LaTeXStrings
using LinearAlgebra, SparseArrays, Statistics, Random
using JLD2
using GraphTransportation

"""
generate 2 random measures on a triangle
compute the geodesic between them with stepsize 1/1000
compare to barycenter computed via gd
"""

function experiment(;
                    n_experiments=1000,
                    h=0.1, maxiters=2^20, tol=1e-10,
                    geo_steps=128, geo_tol=1e-6)
    println("n_experiments: $(n_experiments)\nh:$(h)\nsteps: $(geo_steps)")

    Q, sstate = triangle_markov_chain()
    sqrt_sstate = sqrt.(sstate)

    function random_measure()
        m = rand(size(Q,1))
        m /= sum(m)
        return m ./ sstate
    end

    ρ0 = [2; 0.5; 0.5;]
    ρ1 = [0.5; 0.5; 2.]
    println(ρ0)
    println(ρ1)
    M = cat(ρ0, ρ1, dims=2)
    N = n_experiments
    geovec = discrete_transport(Q, ρ0, ρ1, N=N)

    coords = [[1 - i/N; i/N] for i=1:N-1]

    norm_diffs = zeros(N-1)
    prev_bary = nothing

    for i in 1:N-1
        println("Barycenter $i / $(N-1)")
        bary = barycenter(
            M, coords[i], Q,
            maxiters=maxiters, tol=tol, h=h,
            geodesic_tol=geo_tol, geodesic_steps=geo_steps,
            initialization=prev_bary)
        norm_diffs[i] = norm((geovec.vector.ρ[i + 1,:] - bary) .* sqrt_sstate) / norm(geovec.vector.ρ[i + 1,:] .* sqrt_sstate)
        prev_bary = bary
    end
    @save "DeterministicRecovery_$(geo_steps)_$(geo_tol)_$(tol).jld2" geovec norm_diffs N h maxiters tol geo_steps geo_tol

    return norm_diffs
end

"""
generate n_experiments random measures
pick a random point on the geodesic between those measures
examine recovery
"""
function experiment_randomized(;
                    n_experiments=1000,
                    h=0.01, maxiters=100_000, tol=1e-8,
                    geo_steps=10, geo_tol=1e-11,
                    geodesic_N=100,
                    graph_fn=triangle_markov_chain,
                    mass_floor=nothing,
                    cache="randomized_recovery.jld2")

    Q, sstate = graph_fn()
    sqrt_sstate = sqrt.(sstate)
    N = geodesic_N

    function random_measure()
        m = rand(size(Q,1))
        m /= sum(m)
        ρ = m ./ sstate
        if !isnothing(mass_floor)
            ρ = max.(ρ, mass_floor)
            ρ /= dot(ρ, sstate)
        end
        return ρ
    end

    function random_experiment()
        ρ0 = random_measure()
        ρ1 = random_measure()
        M = cat(ρ0, ρ1, dims=2)
        geovec = discrete_transport(Q, ρ0, ρ1, N=N)
        t = rand(1:N)
        coords = [1 - t/N; t/N]
        bary = barycenter(
            M, coords, Q,
            maxiters=maxiters, tol=tol, h=h,
            geodesic_tol=geo_tol, geodesic_steps=geo_steps,
            uniform_init=true, verbose=false
        )
        return norm((geovec.vector.ρ[t + 1,:] - bary) .* sqrt_sstate) / norm(geovec.vector.ρ[t + 1,:] .* sqrt_sstate)
    end

    # Each trial is independent — parallelize across Julia threads.
    # BLAS threading is useless on the tiny triangle graph and causes contention.
    BLAS.set_num_threads(1)

    norm_diffs = zeros(n_experiments)
    rng_states = Vector{Any}(undef, n_experiments)
    n_done = Threads.Atomic{Int}(0)
    report_every = max(1, n_experiments ÷ 10)

    Threads.@threads for i in 1:n_experiments
        rng_states[i] = copy(Random.default_rng())
        norm_diffs[i] = random_experiment()
        done = Threads.atomic_add!(n_done, 1) + 1
        if done % report_every == 0
            println("Trial $done / $n_experiments")
            flush(stdout)
        end
    end

    graph_name = string(graph_fn)
    @save cache norm_diffs rng_states n_experiments h maxiters tol geo_steps geo_tol geodesic_N graph_name mass_floor
    return norm_diffs
end

function plot_results(norm_diffs; n_experiments=length(norm_diffs))
    f = Figure()
    ax = Axis(f[1,1],
        xlabel=L"\text{Relative error} \; e_{\mathrm{rel}}",
        ylabel=L"\text{Count}",
        title=L"\text{Geodesic recovery} \quad (n = %$(n_experiments))",
    )
    hist!(ax, norm_diffs)
    f
end

function plot_norm_diffs(norm_diffs, N)
    n = length(norm_diffs)
    xs = [i/N for i in 1:n]
    f = Figure()
    ax = Axis(f[1,1],
        xlabel=L"t",
        ylabel=L"\text{Relative error} \; e_{\mathrm{rel}}",
        xticks=(0.1:0.1:1.0, [L"0.%$(i)" for i in 1:10]),
    )
    lines!(ax, xs, norm_diffs)
    f
end

function run_and_plot(N; geo_steps=64, tol=1e-10, geo_tol=1e-6)
    results = experiment(n_experiments=N, geo_steps=geo_steps, geo_tol=geo_tol, tol=tol)
    f = plot_results(results)
    save("deterministic_recovery_$(geo_steps)_$(geo_tol)_$(tol).pdf", f)
    g = plot_norm_diffs(results, N)
    save("deterministic_recovery_norm_diffs_$(geo_steps)_$(geo_tol)_$(tol).pdf", g)
    f, g
end

"""
Run the randomized geodesic recovery experiment and save the figure.
If `cache` points to an existing JLD2 file, loads results from it instead of rerunning.
Pass `force=true` to ignore the cache and rerun.
"""
function run_and_plot_randomized(N;
        geo_steps=10, tol=1e-8, geo_tol=1e-11,
        maxiters=100_000, geodesic_N=100,
        graph_fn=triangle_markov_chain,
        mass_floor=nothing,
        cache="randomized_recovery.jld2",
        pdf="randomized_recovery.pdf",
        force=false)

    if !force && isfile(cache)
        println("Loading from cache: $cache")
        @load cache norm_diffs n_experiments
    else
        norm_diffs = experiment_randomized(
            n_experiments=N, geo_steps=geo_steps, geo_tol=geo_tol,
            tol=tol, maxiters=maxiters, geodesic_N=geodesic_N,
            graph_fn=graph_fn, mass_floor=mass_floor, cache=cache)
        n_experiments = N
    end

    f = plot_results(norm_diffs; n_experiments=n_experiments)
    save(pdf, f)
    println("Saved $pdf")
    f
end

"""
Debug version of experiment_randomized. For each trial, captures:
  - relative error
  - whether barycenter converged (final ‖Δν‖ vs tol)
  - the t value (weight parameter)
  - whether t is near an endpoint (< 5% or > 95%)
  - ρ0, ρ1, the computed barycenter, and the reference geodesic point

Trials with error above `error_threshold` are saved in detail to `debug_trials`.
"""
function experiment_randomized_debug(;
                    n_experiments=200,
                    h=0.01, maxiters=100_000, tol=1e-10,
                    geo_steps=10, geo_tol=1e-11,
                    error_threshold=0.1, mass_floor=0.05)

    Q, sstate = triangle_markov_chain()
    sqrt_sstate = sqrt.(sstate)
    N = 100

    function random_measure()
        m = rand(size(Q,1))
        m /= sum(m)
        ρ = m ./ sstate
        ρ = max.(ρ, mass_floor)
        ρ /= dot(ρ, sstate)
        return ρ
    end

    norm_diffs    = zeros(n_experiments)
    final_norms   = zeros(n_experiments)
    t_values      = zeros(Int, n_experiments)
    converged     = falses(n_experiments)
    rng_states    = Vector{Any}(undef, n_experiments)
    debug_trials  = []

    for i in 1:n_experiments
        i % 10 == 0 && println("Trial $i / $n_experiments")

        rng_states[i] = copy(Random.default_rng())
        ρ0 = random_measure()
        ρ1 = random_measure()
        M  = cat(ρ0, ρ1, dims=2)

        geovec = discrete_transport(Q, ρ0, ρ1, N=N)
        t = rand(1:N)
        t_values[i] = t
        coords = [1 - t/N; t/N]

        ν_bary, nd_hist, _ = barycenter(
            M, coords, Q,
            maxiters=maxiters, tol=tol, h=h,
            geodesic_tol=geo_tol, geodesic_steps=geo_steps,
            return_stats=true, verbose=false
        )

        pos = filter(x -> x > 0, nd_hist)
        last_nd = isempty(pos) ? nd_hist[end] : last(pos)
        final_norms[i] = last_nd
        converged[i]   = last_nd < tol

        ref = geovec.vector.ρ[t + 1, :]
        err = norm((ref - ν_bary) .* sqrt_sstate) / norm(ref .* sqrt_sstate)
        norm_diffs[i] = err

        if err > error_threshold
            push!(debug_trials, (
                trial      = i,
                err        = err,
                t          = t,
                weight     = t/N,
                near_endpt = (t/N < 0.05 || t/N > 0.95),
                converged  = converged[i],
                final_nd   = last_nd,
                ρ0         = copy(ρ0),
                ρ1         = copy(ρ1),
                bary       = copy(ν_bary),
                ref        = copy(ref),
                rng        = rng_states[i],
            ))
            println("\n[DEBUG] Trial $i: err=$(round(err, sigdigits=4)), t=$t (weight=$(round(t/N, digits=3))), converged=$(converged[i]), final_‖Δν‖=$(round(last_nd, sigdigits=3))")
            println("  ρ0   = $(round.(ρ0, sigdigits=4))")
            println("  ρ1   = $(round.(ρ1, sigdigits=4))")
            println("  ref  = $(round.(ref, sigdigits=4))")
            println("  bary = $(round.(ν_bary, sigdigits=4))")
        end
    end

    println("\n=== Summary ===")
    println("  Trials with err > $error_threshold : $(length(debug_trials)) / $n_experiments")
    println("  Trials that did NOT converge       : $(sum(.!converged)) / $n_experiments")
    println("  Errors near endpoints (t/N<5%/>95%): $(isempty(debug_trials) ? 0 : sum(d.near_endpt for d in debug_trials))")
    if !isempty(debug_trials)
        println("  t/N values of bad trials: $(round.([d.weight for d in debug_trials], digits=3))")
        println("  Converged? of bad trials: $([d.converged for d in debug_trials])")
    end

    @save "randomized_recovery_debug.jld2" norm_diffs final_norms t_values converged rng_states debug_trials n_experiments tol geo_steps geo_tol maxiters h mass_floor

    return norm_diffs, final_norms, t_values, converged, debug_trials
end

"""
Run the debug experiment and produce a 4-panel diagnostic figure.
If `force=false` and `randomized_recovery_debug.jld2` exists, loads from cache.
"""
function run_and_plot_debug(N=200;
        geo_steps=10, tol=1e-10, geo_tol=1e-11,
        maxiters=100_000, error_threshold=0.1, mass_floor=0.05,
        cache="randomized_recovery_debug.jld2", force=false)

    if !force && isfile(cache)
        println("Loading from cache: $cache")
        @load cache norm_diffs final_norms t_values converged n_experiments tol
    else
        norm_diffs, final_norms, t_values, converged, _ =
            experiment_randomized_debug(n_experiments=N, geo_steps=geo_steps, geo_tol=geo_tol,
                                        tol=tol, maxiters=maxiters, error_threshold=error_threshold,
                                        mass_floor=mass_floor)
        n_experiments = N
    end

    f = Figure(size=(900, 700))

    ax1 = Axis(f[1,1],
        xlabel=L"\text{Relative error} \; e_{\mathrm{rel}}",
        ylabel=L"\text{Count}",
        title=L"\text{Error distribution} \quad (n = %$(n_experiments))",
    )
    hist!(ax1, norm_diffs)

    ax2 = Axis(f[1,2],
        xlabel=L"t/N",
        ylabel=L"\text{Relative error} \; e_{\mathrm{rel}}",
        title=L"\text{Error vs.\ interpolation parameter}",
    )
    scatter!(ax2, t_values ./ 100, norm_diffs, markersize=5,
             color=ifelse.(converged, :steelblue, :red))

    ax3 = Axis(f[2,1],
        xlabel=L"\log_{10}(\Vert\Delta\nu\Vert)",
        ylabel=L"\text{Count}",
        title=L"\text{Convergence check} \quad (\text{dashed} = \log_{10}(\varepsilon))",
    )
    hist!(ax3, log10.(clamp.(final_norms, 1e-15, Inf)))
    vlines!(ax3, [log10(tol)], color=:red, linestyle=:dash)

    ax4 = Axis(f[2,2],
        xlabel=L"\log_{10}(\Vert\Delta\nu\Vert)",
        ylabel=L"\text{Relative error} \; e_{\mathrm{rel}}",
        title=L"\text{Error vs.\ convergence quality}",
    )
    scatter!(ax4, log10.(clamp.(final_norms, 1e-15, Inf)), norm_diffs, markersize=5)

    save("randomized_recovery_debug.pdf", f)
    println("Saved randomized_recovery_debug.pdf")
    f
end
