# MeansComparison.jl
#
# The same three reference measures and the same equal weights, three transport
# metrics: the discrete transport barycenter under the geometric, harmonic and (exact)
# logarithmic means on the 49-node USA graph, computed by Riemannian descent
# with Hamiltonian-shooting geodesics (`method=:shooting`), which is exact in time and
# takes the logarithmic mean as is. The mean is a property of the geometry, so each
# barycenter is computed on `MarkovGraph(G; mean=θ)`. The barycenter panels are ordered
# by the variance J. (The joint SOCP handles the geometric and harmonic means at N=32
# in a few seconds each, but its quadrature-log program stalls in Clarabel for N ≥ 8 on
# this graph, which the library reports as an error rather than returning a stale
# iterate.) The arithmetic mean is left out of the figure to keep a 3×2 layout; adding
# ("arithmetic", ArithmeticMean()) to `means` restores it.
#
# Also reports J = Σλᵢ W²(refᵢ, ν) for each mean (which orders like the means:
# harmonic ≥ geometric ≥ logarithmic), the pairwise L²(u) distances between the
# barycenters, and the pairwise distance W(ref₁, ref₂) under each mean.
#
# Run from src/experiments/ with --project=.  Writes means_comparison.{jld2,pdf,png};
# the PNG is the figure used on the documentation index page.

include("../core/CommonGraphs.jl")
include("./ExperimentUtils.jl")
using GraphTransportation
using GraphMakie, Graphs, CairoMakie
using LinearAlgebra, Printf
using JLD2

Q, u, geo_cx, geo_cy = load_usa_mc()
A = Q .> 0
n = size(Q, 1)
G0 = MarkovGraph(Q, u)
refs = [random_geographic_concentration(A, weight=10, center=c) ./ u for c in (34, 28, 4)]   # AZ, IL, VA
λ = fill(1 / 3, 3)
TOL = 1e-5      # descent stops when the metric norm of the Riemannian gradient is below this
means = [("geometric", GeometricMean()), ("harmonic", HarmonicMean()), ("logarithmic", LogarithmicMean())]
norm_u(v) = sqrt(sum(u .* v .^ 2))       # L²(u) norm, u the stationary distribution

const CACHE = "means_comparison.jld2"
if isfile(CACHE)
    d = load(CACHE); bars = d["bars"]; Js = d["Js"]; W12 = d["W12"]; times = d["times"]
else
    bars = Dict{String,Vector{Float64}}(); Js = Dict{String,Float64}(); W12 = Dict{String,Float64}(); times = Dict{String,Float64}()
    for (name, θ) in means
        G = MarkovGraph(G0; mean=θ)
        t = @elapsed (ν, J, info) = barycenter(G, refs, λ; method=:shooting, tol=TOL)
        bars[name] = ν; Js[name] = J; times[name] = t
        W12[name] = transport_cost(G, refs[1], refs[2]; method=:shooting)
        @printf("%-12s J=%.5f  W(ref1,ref2)=%.4f  min ν=%.2e  %d iters, status=%s  (%.1fs)\n",
                name, J, W12[name], minimum(ν), info.iters, info.status, t)
    end
    @save CACHE bars Js W12 times
end

println("\npairwise ‖ν_a − ν_b‖_{L²(u)} between the barycenters:")
for i in 1:length(means), j in i+1:length(means)
    a, b = means[i][1], means[j][1]
    @printf("  %-18s vs %-18s %.4f\n", a, b, norm_u(bars[a] .- bars[b]))
end

# ---- figure: row 1 the three references, row 2 the three barycenters, shared color scale ----
g_plot = SimpleGraph(n)
for i in 1:n, j in i+1:n
    A[i, j] != 0 && add_edge!(g_plot, i, j)
end
positions = [Point2f(geo_cx[i], geo_cy[i]) for i in 1:n]
refs_disp = [r .* u for r in refs]
order = sort(1:length(means); by=k -> Js[means[k][1]])          # ascending variance J = Σλᵢ W²(refᵢ, ν)
means_sorted = means[order]
bars_disp = [bars[name] .* u for (name, _) in means_sorted]
vmin, vmax = 0.0, maximum(vcat(vcat(refs_disp...), vcat(bars_disp...)))
cmap = :plasma

fig = Figure(size=(1500, 640), backgroundcolor=:white)
function panel!(pos, data, title)
    ax = Axis(fig[pos...], title=title, titlesize=22, aspect=DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    graphplot!(ax, g_plot; layout=(_) -> positions, node_color=data, node_size=24,
               node_attr=(colormap=cmap, colorrange=(vmin, vmax)), edge_color=(:gray60, 0.5), edge_width=1.5)
end
for (k, r) in enumerate(refs_disp)
    panel!((1, k), r, L"\nu_%$k")
end
for (k, (name, _)) in enumerate(means_sorted)
    panel!((2, k), bars_disp[k], @sprintf("%s mean\nJ = %.4f", name, Js[name]))
end
Colorbar(fig[1:2, 4], colormap=cmap, limits=(vmin, vmax), label="mass", labelsize=18, height=Relative(0.7))
Label(fig[1, 0], "references", rotation=π/2, fontsize=20, tellheight=false)
Label(fig[2, 0], "barycenters (by J, ascending)", rotation=π/2, fontsize=20, tellheight=false)
Label(fig[0, :], "Equal-weight barycenter of three reference measures on the USA graph, one transport metric per panel (Hamiltonian shooting)",
      fontsize=20, tellwidth=false)
save("means_comparison.pdf", fig)
save("means_comparison.png", fig; px_per_unit=1)
println("Saved means_comparison.{jld2,pdf,png}")
