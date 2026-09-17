# SynthesisAnalysisMismatch.jl
#
# Why the synthesis and analysis pipelines are not interchangeable between methods.
#
# A barycenter is a stationary point of Σᵢ λᵢ W²(ν, refᵢ) *in the discretization that
# produced it*. Analyzing it (recovering λ from a Gram matrix of tangent vectors at ν)
# only recovers the true λ to solver tolerance when the analysis checks stationarity
# in that same convention; otherwise it recovers λ only to the discretization error.
# This script makes that visible on the MA house graph by crossing two target
# points with two Gram-matrix conventions and two time resolutions:
#   targets:     ν_socp (from barycenter_socp, N=2) and bar_descent (from the
#                Chambolle-Pock descent, N=2), reloaded from
#                ma_house_socp_comparison.jld2 (bar_descent alone took ~58 min)
#   conventions: analyze_socp with convention=:potential (endpoint-potential Gram,
#                the SOCP's own stationarity) and convention=:momentum (initial
#                momentum, the descent's stationarity)
#   N:           2 (matching synthesis) and 10 (mismatched)
# Expected pattern: each target is recovered essentially exactly by the convention
# and N it was synthesized in, and only approximately by the other; refining N
# shrinks but does not remove the mismatch. The residual-ratio diagnostic
# λᵀAλ / λ̂ᵀAλ̂ (≈ 1 when the true λ is already the QP minimizer) is reported for each
# cell.
#
# Run from src/experiments/ with --project=.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using LinearAlgebra, Printf
using Shapefile, LibGEOS
using JLD2

Q_unw, u_unw = ma_house_markov_chain()
n_nodes = size(Q_unw, 1)
A = Q_unw .> 0
centers = [5, 50, 100, 145]
refs = [random_geographic_concentration(A, weight=10, center=c) ./ u_unw for c in centers]
λ = [0.4, 0.3, 0.2, 0.1]
G = MarkovGraph(Q_unw, u_unw)

d = load("ma_house_socp_comparison.jld2")
ν_socp = d["ν_socp"]; bar_descent = d["bar_descent"]
rc_descent_cached = d["rc_descent"]; rc_socp_cached = d["rc_socp"]

relerr(x) = norm(λ .- x) / norm(λ)
condA(A) = (e = abs.(eigvals(A)); maximum(e) / minimum(e))

println("cached recovery values (each from its own synthesis convention), for reference:")
@printf("  descent point via analysis():     %s  rel.err %.4f\n", round.(rc_descent_cached; digits=4), relerr(rc_descent_cached))
@printf("  ν_socp via analyze_socp (old):    %s  rel.err %.4f\n", round.(rc_socp_cached; digits=4), relerr(rc_socp_cached))
println()

results = Dict{Tuple{String,Symbol,Int},NamedTuple}()
for (name, target) in (("ν_socp", ν_socp), ("bar_descent", bar_descent)),
    N in (2, 10),
    convention in (:potential, :momentum)
    local λ̂, Amat
    t = @elapsed redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            (λ̂_, Amat_) = analysis(G, target, refs; N=N, convention=convention, return_system=true)
            λ̂ = vec(λ̂_); Amat = Amat_
        end
    end
    true_res = λ' * Amat * λ
    qp_res   = λ̂' * Amat * λ̂
    results[(name, convention, N)] = (; λ̂, err=relerr(λ̂), true_res, qp_res, ratio=true_res / qp_res,
                                         diagA=diag(Amat), cond=condA(Amat), t)
    @printf("%-12s N=%-3d %-10s λ̂=%s  rel.err=%.4f  λᵀAλ=%.4g  λ̂ᵀAλ̂=%.4g  ratio=%.3f  max diag(A)=%.4g  cond=%.3g  (%.1fs)\n",
            name, N, convention, round.(λ̂; digits=4), relerr(λ̂), true_res, qp_res, true_res / qp_res,
            maximum(diag(Amat)), condA(Amat), t)
end

@save "synthesis_analysis_mismatch.jld2" results
println("Saved synthesis_analysis_mismatch.jld2")
