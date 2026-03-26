using QuadGK
using LinearAlgebra
using Printf
using CairoMakie
using LaTeXStrings
using GraphTransportation
using JLD2

# ── Two-point graph setup ─────────────────────────────────────────────────────
# Q = [[0 1],[1 0]], π = [1/2, 1/2].
# Probability densities w.r.t. π are parameterised by r ∈ [-1,1]:
#   ρ(r) = [1-r, 1+r]   (ρ_a(r) = 1-r, ρ_b(r) = 1+r)
# so ρ(-1) = [2,0] (all mass at a) and ρ(1) = [0,2] (all mass at b).
#
# For the geometric mean θ_geom(s,t) = √(st) the distance formula is
#
#   W(ρ(α), ρ(β)) = (1/√2) ∫_α^β (1-r²)^{-1/4} dr
#
# and the geodesic is ρ(γ(t)) where γ satisfies ODE (37):
#
#   γ'(t) = C·√2·((1-γ)(1+γ))^{1/4},   γ(0) = α
#
# with C = W(ρ(α), ρ(β)).  We compare:
#   • discrete_transport (geometric mean, Chambolle–Pock, N_dt time steps)
#   • barycenter (WGD, one call per time point)
#   • explicit Euler on the ODE (N_ode = 2000 steps)
#   • arithmetic averaging (diagonal/linear interpolation) — not admissible

Q = [0.0 1.0; 1.0 0.0]

# Small regularisation so that (i) the boundary densities are strictly positive
# (avoiding exact zeros that stall explicit Euler) and (ii) quadgk can
# integrate without hitting the integrable-but-singular boundary values ±1.
const ε_reg = 1e-4
const α = -1.0 + ε_reg
const β =  1.0 - ε_reg

ρ_A = [1.0 - α, 1.0 + α]   # ≈ [2, 0]
ρ_B = [1.0 - β, 1.0 + β]   # ≈ [0, 2]

# ── Analytical distance (numerical quadrature) ────────────────────────────────
function two_point_distance(s, t)
    val, _ = quadgk(r -> (1 - r^2)^(-1/4), s, t)
    return val / sqrt(2)
end

# ── ODE explicit Euler (N_ode = 2000 steps) ───────────────────────────────────
function solve_ode_euler(a, b; N=2000)
    C = two_point_distance(a, b)
    h = 1.0 / N
    γ = Vector{Float64}(undef, N + 1)
    γ[1] = a
    for i in 1:N
        x = γ[i]
        γ[i+1] = clamp(x + h * C * sqrt(2) * max(0.0, (1 - x) * (1 + x))^(1/4), a, b)
    end
    # Linear interpolant so we can query any τ ∈ [0,1]
    function interp(τ)
        raw  = τ * N
        lo   = clamp(floor(Int, raw), 0, N - 1)
        frac = raw - lo
        γ[lo + 1] * (1 - frac) + γ[lo + 2] * frac
    end
    return interp
end

const N_ode = 2000
const N_dt  = 1000

if isfile("erbar_ode.jld2")
    println("Loading saved data from erbar_ode.jld2 …")
    @load "erbar_ode.jld2" ts ρ_b_num ρ_b_bary ρ_b_ode ρ_b_arith
else
    println("Solving ODE (explicit Euler, N=$N_ode) …")
    ode_γ = solve_ode_euler(α, β; N=N_ode)

    # ── Discrete transport (geometric mean, Chambolle–Pock) ───────────────────
    println("Running discrete_transport (N=$N_dt) …")
    geo = discrete_transport(Q, ρ_A, ρ_B; N=N_dt, tol=1e-10, progress=true)

    ts = range(0.0, 1.0, length=N_dt + 1)

    ρ_b_num   = geo.vector.ρ[:, 2]
    ρ_b_ode   = [1.0 + ode_γ(t) for t in ts]
    ρ_b_arith = [(1 - t) * ρ_A[2] + t * ρ_B[2] for t in ts]

    # ── Barycenter (WGD, one call per interior time point) ───────────────────
    M = hcat(ρ_A, ρ_B)

    println("Running barycenter for $(length(ts)) time points …")
    ρ_b_bary = Vector{Float64}(undef, length(ts))
    for (i, t) in enumerate(ts)
        if t == 0.0
            ρ_b_bary[i] = ρ_A[2]
        elseif t == 1.0
            ρ_b_bary[i] = ρ_B[2]
        else
            w = [1.0 - t, t]
            bary = redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    barycenter(M, w, Q;
                               tol=1e-10, geodesic_tol=1e-10, geodesic_steps=50,
                               verbose=false)
                end
            end
            ρ_b_bary[i] = bary[2]
        end
        if i % 10 == 0 || i == length(ts)
            @printf("  %d / %d done\n", i, length(ts))
        end
    end

    @save "erbar_ode.jld2" ts ρ_b_num ρ_b_bary ρ_b_ode ρ_b_arith α β N_dt N_ode
    println("Data saved to erbar_ode.jld2")
end

# ── Precompute diffs ──────────────────────────────────────────────────────────
diff_dt   = ρ_b_num  .- ρ_b_ode
diff_bary = ρ_b_bary .- ρ_b_ode

# ── Figure 1: geodesic curves ─────────────────────────────────────────────────
fig1 = Figure(size=(600, 480))
ax1 = Axis(fig1[1, 1];
    xlabel = L"t",
    ylabel = L"\rho_b(t)",
    title  = L"\text{Mass at node } b \text{ along geodesic}",
    xticks = 0:0.2:1,
)
lines!(ax1, ts, ρ_b_arith; color=:black, linewidth=1.5, label=L"\text{Arithmetic (diagonal)}")
lines!(ax1, ts, ρ_b_ode;   color=:gray,  linewidth=1.5, linestyle=:dash,
       label=latexstring("\\text{ODE Euler } (N=$(N_ode))"))
lines!(ax1, ts, ρ_b_num;   color=:green, linewidth=2,
       label=latexstring("\\text{discrete transport } (N=$(N_dt))"))
lines!(ax1, ts, ρ_b_bary;  color=:blue,  linewidth=2, label=L"\text{barycenter (WGD)}")
axislegend(ax1; position=:lt, framevisible=false)
save("erbar_ode.pdf", fig1)
println("Figure saved to erbar_ode.pdf")

# ── Figure 2: error curves ────────────────────────────────────────────────────
fig2 = Figure(size=(600, 480))
ax2 = Axis(fig2[1, 1];
    xlabel = L"t",
    ylabel = L"\rho_b^{\mathrm{num}}(t) - \rho_b^{\mathrm{ODE}}(t)",
    title  = L"\text{Difference: numerical} - \text{ODE Euler}",
    xticks = 0:0.2:1,
)
hlines!(ax2, [0.0]; color=:black, linewidth=0.8, linestyle=:dot)
lines!(ax2, ts, diff_dt;   color=:green, linewidth=2,
       label=L"\text{discrete transport} - \text{ODE}")
lines!(ax2, ts, diff_bary; color=:blue,  linewidth=2,
       label=L"\text{barycenter} - \text{ODE}")
axislegend(ax2; position=:lt, framevisible=false)
save("erbar_ode_errors.pdf", fig2)
println("Figure saved to erbar_ode_errors.pdf")

# ── Figure 3: discrete transport - ODE ───────────────────────────────────────
fig3 = Figure(size=(600, 480))
ax3 = Axis(fig3[1, 1];
    xlabel = L"t",
    ylabel = L"\rho_b^{\mathrm{dt}}(t) - \rho_b^{\mathrm{ODE}}(t)",
    title  = L"\text{discrete transport} - \text{ODE Euler}",
    xticks = 0:0.2:1,
)
hlines!(ax3, [0.0]; color=:black, linewidth=0.8, linestyle=:dot)
lines!(ax3, ts, diff_dt; color=:green, linewidth=2)
save("erbar_ode_diff_dt.pdf", fig3)
println("Figure saved to erbar_ode_diff_dt.pdf")

# ── Figure 4: barycenter - ODE ────────────────────────────────────────────────
fig4 = Figure(size=(600, 480))
ax4 = Axis(fig4[1, 1];
    xlabel = L"t",
    ylabel = L"\rho_b^{\mathrm{bary}}(t) - \rho_b^{\mathrm{ODE}}(t)",
    title  = L"\text{barycenter} - \text{ODE Euler}",
    xticks = 0:0.2:1,
)
hlines!(ax4, [0.0]; color=:black, linewidth=0.8, linestyle=:dot)
lines!(ax4, ts, diff_bary; color=:blue, linewidth=2)
save("erbar_ode_diff_bary.pdf", fig4)
println("Figure saved to erbar_ode_diff_bary.pdf")

# ── Figure 5: histogram discrete transport ────────────────────────────────────
fig5 = Figure(size=(600, 480))
ax5 = Axis(fig5[1, 1];
    xlabel = L"|\rho_b^{\mathrm{dt}}(t) - \rho_b^{\mathrm{ODE}}(t)|",
    ylabel = L"\text{Count}",
    title  = L"\text{Distribution: } |\text{discrete transport} - \text{ODE Euler}|",
)
hist!(ax5, abs.(diff_dt); bins=30, color=(:green, 0.75))
save("erbar_ode_hist_dt.pdf", fig5)
println("Figure saved to erbar_ode_hist_dt.pdf")

# ── Figure 6: histogram barycenter ────────────────────────────────────────────
fig6 = Figure(size=(600, 480))
ax6 = Axis(fig6[1, 1];
    xlabel = L"|\rho_b^{\mathrm{bary}}(t) - \rho_b^{\mathrm{ODE}}(t)|",
    ylabel = L"\text{Count}",
    title  = L"\text{Distribution: } |\text{barycenter} - \text{ODE Euler}|",
)
hist!(ax6, abs.(diff_bary); bins=30, color=(:blue, 0.75))
save("erbar_ode_hist_bary.pdf", fig6)
println("Figure saved to erbar_ode_hist_bary.pdf")
