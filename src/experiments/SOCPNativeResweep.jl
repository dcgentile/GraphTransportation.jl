# SOCPNativeResweep.jl
#
# Rerun the SOCPGridConsistency.jl (24
# cells: k×k grids, k=5..10, N∈{2,5,10,20}) and SOCPProximityStudy.jl (32 cells:
# k∈{7,10}, reference-pair separation d, N∈{2,5,10,20}) round-trip sweeps, but
# analyze every synthesized barycenter with BOTH Gram-matrix conventions:
#   :momentum  — the original analyze_socp (borrowed from CP's analysis())
#   :potential — the native, endpoint-potential Gram matrix (the Riemannian inner product)
# Same deterministic references/λ as the two original scripts. Barycenters are
# stored in the cache this time so future re-analyses need no resynthesis.
#
# Run from src/experiments/ with --project=.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using LinearAlgebra, Printf
using CairoMakie
using JLD2

const λ  = [0.4, 0.3, 0.2, 0.1]
const Ns = [2, 5, 10, 20]

# --- cell definitions, copied from the two original scripts ---
grid_corners(k) = [1, k, k * (k - 1) + 1, k * k]
prox_dists(k)   = unique(round.(Int, (k - 1) .* [1.0, 2/3, 1/3, 1/(k - 1)]))
prox_centers(k, d) = [1, d + 1, (k - 1) * k + 1, k * k]

function build(k, centers)
    Q, π = grid_markov_chain(k)
    A = Q .> 0
    refs = [random_geographic_concentration(A, weight=10, center=c) ./ π for c in centers]
    return MarkovGraph(Q, π), refs
end

# (sweep, k, d, N) -> stats; d = k-1 for the grid sweep (corner-to-corner)
cells = vcat([(:grid, k, k - 1) for k in 5:10],
             [(:prox, k, d) for k in (7, 10) for d in prox_dists(k)])

relerr(x) = norm(λ .- x) / norm(λ)
condA(A) = (e = abs.(eigvals(A)); maximum(e) / minimum(e))

function analyze_both(G, ν, refs, N)
    out = Dict{Symbol,NamedTuple}()
    for convention in (:momentum, :potential)
        local λ̂, Amat
        t = @elapsed redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                (λ̂_, Amat_) = analyze_socp(G, ν, refs; N=N, convention=convention, return_system=true)
                λ̂ = vec(λ̂_); Amat = Amat_
            end
        end
        out[convention] = (; λ̂, err=relerr(λ̂), ratio=(λ' * Amat * λ) / (λ̂' * Amat * λ̂),
                             true_res=λ' * Amat * λ, cond=condA(Amat), t)
    end
    return out
end

let
    G_warm = MarkovGraph([0.0 1.0; 1.0 0.0], [0.5, 0.5])
    geodesic_socp(G_warm, [1.5, 0.5], [0.5, 1.5]; N=2)
    barycenter_socp(G_warm, [[1.5, 0.5], [0.5, 1.5]], [0.5, 0.5]; N=2)
end

const CACHE = "socp_native_resweep.jld2"

if isfile(CACHE)
    println("Loading cached results from $CACHE ...")
    results = load(CACHE, "results")
else
    results = Dict{Tuple{Symbol,Int,Int,Int},NamedTuple}()
    for (sweep, k, d) in cells
        centers = sweep == :grid ? grid_corners(k) : prox_centers(k, d)
        println("$(sweep) k=$k (V=$(k*k)) d=$d centers=$centers")
        G, refs = build(k, centers)
        for N in Ns
            t_synth = @elapsed (ν, J, _) = barycenter_socp(G, refs, λ; N=N, silent=true)
            both = analyze_both(G, ν, refs, N)
            @printf("  N=%-3d J=%.4g  momentum: err=%.3e ratio=%.3f | potential: err=%.3e ratio=%.3f λᵀAλ=%.2e  (synth %.1fs)\n",
                    N, J, both[:momentum].err, both[:momentum].ratio,
                    both[:potential].err, both[:potential].ratio, both[:potential].true_res, t_synth)
            results[(sweep, k, d, N)] = (; sweep, k, d, N, V=k * k, ν, J, t_synth,
                                           momentum=both[:momentum], potential=both[:potential])
        end
    end
    @save CACHE results
end

println("\n== Summary ==")
@printf("%-5s %-4s %-4s %-4s %-12s %-12s %-10s %-10s\n", "sweep", "V", "d", "N", "err(momentum)", "err(potential)", "ratio(mom)", "ratio(pot)")
for (sweep, k, d) in cells, N in Ns
    r = results[(sweep, k, d, N)]
    @printf("%-5s %-4d %-4d %-4d %-12.3e %-12.3e %-10.3f %-10.3f\n", sweep, r.V, d, N,
            r.momentum.err, r.potential.err, r.momentum.ratio, r.potential.ratio)
end
for conv in (:momentum, :potential)
    errs = [getfield(results[(s, k, d, N)], conv).err for (s, k, d) in cells, N in Ns]
    @printf("%-10s worst rel.err over all %d cells: %.3e   median: %.3e\n", conv, length(errs), maximum(errs), sort(vec(errs))[end ÷ 2 + 1])
end

# --- Figure: grid sweep (err vs V, per N) and proximity sweep (err vs d, per N), old vs new ---
fig = Figure(size=(1400, 900))
for (row, conv) in enumerate((:momentum, :potential))
    ax = Axis(fig[row, 1], title="grid sweep — $(conv) Gram matrix", xlabel="V", ylabel="‖λ-λ̂‖/‖λ‖", yscale=log10)
    for N in Ns
        ks = 5:10
        errs = [max(getfield(results[(:grid, k, k - 1, N)], conv).err, 1e-16) for k in ks]
        scatterlines!(ax, ks .^ 2, errs; label="N=$N")
    end
    axislegend(ax; position=:rb)
    for (col, k) in enumerate((7, 10))
        ax2 = Axis(fig[row, col + 1], title="proximity V=$(k*k) — $(conv)", xlabel="separation d", ylabel="‖λ-λ̂‖/‖λ‖",
                   yscale=log10, xreversed=true)
        for N in Ns
            ds = prox_dists(k)
            errs = [max(getfield(results[(:prox, k, d, N)], conv).err, 1e-16) for d in ds]
            scatterlines!(ax2, ds, errs; label="N=$N")
        end
        axislegend(ax2; position=:rb)
    end
end
Label(fig[0, :], "SOCP round-trip recovery error: momentum-based (top) vs endpoint-potential (bottom) Gram matrix",
      fontsize=15, tellwidth=false)
save("socp_native_resweep.pdf", fig)
println("Saved socp_native_resweep.{jld2,pdf}")
