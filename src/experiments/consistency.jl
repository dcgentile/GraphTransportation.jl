# consistency.jl
#
# Analysis pipeline validation: synthesize barycenters with the SOCP at a known λ and
# recover λ with every analysis backend, sweeping the synthesis resolution N and, for
# the shooting backend, the integrator step count. Expected pattern (see
# SynthesisAnalysisMismatch.jl for the mechanism): recovery is exact at the synthesis
# N for method=:socp, and O(1/N) in the synthesis N for :shooting and :chambolle_pock,
# because each backend checks stationarity in its own discrete convention.
#
#   graphs:  4x4 grid (16), 7x7 grid (49); 4 references at the corners, λ=[0.4,0.3,0.2,0.1]
#   synth:   barycenter(...; N) for N ∈ {2,5,10,20,40}
#   analyze: :socp at the same N; :socp at N=10 (mismatched); :shooting at
#            nsteps ∈ {20, 150}; :chambolle_pock at N=10
#   reports: ‖λ̂-λ‖/‖λ‖ per cell, the Gram residual of the true λ, and wall-clock
#
# Run from src/experiments/ with --project=.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using LinearAlgebra, Statistics, Printf
using CairoMakie
using JLD2

const λ  = [0.4, 0.3, 0.2, 0.1]
const Ns = [2, 5, 10, 20, 40]
const Ks = [4, 7]
corners(k) = [1, k, k * (k - 1) + 1, k * k]

function build(k)
    Q, π = grid_markov_chain(k)
    A = Q .> 0
    refs = [random_geographic_concentration(A, weight=10, center=c) ./ π for c in corners(k)]
    MarkovGraph(Q, π), refs
end
relerr(x) = norm(λ .- vec(x)) / norm(λ)

backends = [
    ("socp, same N",        (G, ν, refs, N) -> analysis(G, ν, refs; N=N)),
    ("socp, N=10",          (G, ν, refs, N) -> analysis(G, ν, refs; N=10)),
    ("shooting, nsteps=20", (G, ν, refs, N) -> analysis(G, ν, refs; method=:shooting, nsteps=20)),
    ("shooting, nsteps=150",(G, ν, refs, N) -> analysis(G, ν, refs; method=:shooting, nsteps=150)),
    ("chambolle_pock, N=10",(G, ν, refs, N) -> analysis(G, ν, refs; method=:chambolle_pock, N=10, tol=1e-10)),
]

let (G, refs) = build(4)   # JIT warm-up
    ν, _, _ = barycenter(G, refs, λ; N=2)
    for (_, f) in backends; redirect_stdout(devnull) do; f(G, ν, refs, 2); end; end
end

const CACHE = "consistency.jld2"
if isfile(CACHE)
    results = load(CACHE, "results")
else
    results = Dict{Tuple{Int,Int,String},NamedTuple}()
    for k in Ks
        G, refs = build(k)
        println("== $(k)x$(k) grid ==")
        for N in Ns
            ν, J, _ = barycenter(G, refs, λ; N=N)
            for (name, f) in backends
                t = @elapsed λ̂ = redirect_stdout(devnull) do; redirect_stderr(devnull) do; f(G, ν, refs, N); end; end
                results[(k, N, name)] = (; err=relerr(λ̂), t)
                @printf("  N=%-3d %-22s err=%.3e  t=%.2fs\n", N, name, relerr(λ̂), t)
            end
        end
    end
    @save CACHE results
end

fig = Figure(size=(1200, 500))
for (col, k) in enumerate(Ks)
    ax = Axis(fig[1, col], title="$(k)x$(k) grid (V=$(k*k))", xlabel="synthesis N", ylabel="‖λ̂-λ‖/‖λ‖",
              xscale=log10, yscale=log10)
    for (name, _) in backends
        scatterlines!(ax, Ns, [max(results[(k, N, name)].err, 1e-16) for N in Ns]; label=name)
    end
    col == 2 && axislegend(ax; position=:rt)
end
Label(fig[0, :], "Coordinate recovery vs. synthesis resolution, by analysis backend (SOCP-synthesized barycenters)", fontsize=15, tellwidth=false)
save("consistency.pdf", fig)
println("Saved consistency.{jld2,pdf}")
