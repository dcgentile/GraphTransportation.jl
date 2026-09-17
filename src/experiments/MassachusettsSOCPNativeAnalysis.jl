# MassachusettsSOCPNativeAnalysis.jl
#
# Re-analyze the cached MA house SOCP
# barycenter (ν_socp from MassachusettsSOCPComparison.jl, N=2) with the
# potential-based ("native") Gram matrix, and compare against the momentum-based
# Gram matrix that produced the 27.6% recovery error. Also reports the
# residual-ratio diagnostic λᵀAλ / λ̂ᵀAλ̂ for both conventions and both target
# points (ν_socp and bar_descent).
#
# Nothing is resynthesized: ν_socp and bar_descent are reloaded from
# ma_house_socp_comparison.jld2 (bar_descent alone took ~58 min).
#
# Run from src/experiments/ with --project=.

include("../CommonGraphs.jl")
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

println("cached (momentum-based) recovery, for reference:")
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
            (λ̂_, Amat_) = analyze_socp(G, target, refs; N=N, convention=convention, return_system=true)
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

@save "ma_house_socp_native_analysis.jld2" results
println("Saved ma_house_socp_native_analysis.jld2")
