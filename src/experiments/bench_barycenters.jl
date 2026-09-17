# bench_barycenters.jl
#
# Barycenter synthesis side by side: the joint SOCP (global optimum, one solve) vs. the
# paper's intrinsic gradient descent with Chambolle-Pock geodesics, on graphs of
# increasing size. Reports wall-clock, the objective J(ν) = Σλᵢ W²(refᵢ, ν) evaluated
# for both barycenters by the same solver (SOCP geodesics at N=20) so the optimality
# certificate J_socp ≤ J_descent is meaningful, the π-norm discrepancy between the two
# barycenters, and coordinate recovery of each by its own analysis backend.
# (Entropic/Sinkhorn barycenters are a different object - a design note on whether and
# how to fold them into the method= API is pending - and are not included here.)
#
#   graphs: hypercube (16), 5x5 grid (25), 7x7 grid (49), 10x10 grid (100)
#   refs:   4 geographically concentrated measures, λ=[0.4,0.3,0.2,0.1]
#   socp:   N=10;  descent: h=0.1, maxiters=300, geodesic_steps=10, geodesic_tol=1e-8
#
# Run from src/experiments/ with --project=.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using LinearAlgebra, Statistics, Printf
using JLD2

const λ = [0.4, 0.3, 0.2, 0.1]
norm_pi(v, π) = sqrt(sum(π .* v .^ 2))
relerr(x) = norm(λ .- vec(x)) / norm(λ)

function refs_for(Q, π, centers)
    A = Q .> 0
    [random_geographic_concentration(A, weight=10, center=c) ./ π for c in centers]
end
cases = [
    ("hypercube (16)",  weighted_hypercube_markov_chain()..., [1, 6, 11, 16]),
    ("grid 5x5 (25)",   grid_markov_chain(5)...,  [1, 5, 21, 25]),
    ("grid 7x7 (49)",   grid_markov_chain(7)...,  [1, 7, 43, 49]),
    ("grid 10x10 (100)", grid_markov_chain(10)..., [1, 10, 91, 100]),   # descent: ~64 min
]

let (_, Q, π, c) = cases[1]; G = MarkovGraph(Q, π); refs = refs_for(Q, π, c)   # JIT warm-up
    barycenter(G, refs, λ; N=2)
    barycenter(G, refs, λ; method=:chambolle_pock, h=0.1, maxiters=3, geodesic_steps=2, geodesic_tol=1e-6, verbose=false)
end

const CACHE = "bench_barycenters.jld2"
if isfile(CACHE)
    results = load(CACHE, "results")
else
    results = Dict{String,NamedTuple}()
    for (name, Q, π, centers) in cases
        G = MarkovGraph(Q, π); refs = refs_for(Q, π, centers)
        println("== $name ==")
        t_socp = @elapsed (ν_socp, J_socp, _) = barycenter(G, refs, λ; N=10)
        t_cp   = @elapsed (ν_cp, _, info_cp) = barycenter(G, refs, λ; method=:chambolle_pock, h=0.1, maxiters=300,
                                                          geodesic_steps=10, geodesic_tol=1e-8, verbose=false)
        J_at(ν) = sum(λ[i] * geodesic(G, refs[i], ν; N=20).W2 for i in eachindex(refs))
        Js, Jc = J_at(ν_socp), J_at(ν_cp)
        rc_socp = relerr(redirect_stdout(devnull) do; analysis(G, ν_socp, refs; N=10); end)
        rc_cp   = relerr(redirect_stdout(devnull) do; analysis(G, ν_cp, refs; method=:chambolle_pock, N=10, tol=1e-8); end)
        iters_cp = something(findlast(!=(0.0), info_cp.variances), length(info_cp.variances))
        results[name] = (; t_socp, t_cp, J_socp=Js, J_cp=Jc, certificate=(Js <= Jc + 1e-9), disc=norm_pi(ν_socp .- ν_cp, π), rc_socp, rc_cp, iters_cp)
        @printf("  socp %.2fs  descent %.1fs (%d iters)  J: socp=%.5f descent=%.5f cert=%s  ‖Δν‖_π=%.3e  rc.err socp=%.1e descent=%.1e\n",
                t_socp, t_cp, iters_cp, Js, Jc, Js <= Jc + 1e-9, norm_pi(ν_socp .- ν_cp, π), rc_socp, rc_cp)
    end
    @save CACHE results
end

println("\n== summary ==")
@printf("%-18s %8s %9s %10s %10s %6s %10s %9s %9s\n", "graph", "t_socp", "t_desc", "J_socp", "J_desc", "cert", "‖Δν‖_π", "rc_socp", "rc_desc")
for (name, _, _, _) in cases
    r = results[name]
    @printf("%-18s %8.2f %9.1f %10.5f %10.5f %6s %10.3e %9.1e %9.1e\n", name, r.t_socp, r.t_cp, r.J_socp, r.J_cp, r.certificate, r.disc, r.rc_socp, r.rc_cp)
end
println("Saved bench_barycenters.jld2")
