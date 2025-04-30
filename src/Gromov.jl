include("EarthMover.jl")
using CairoMakie
using LinearAlgebra

function gromov_convergence(N, n)
    Q = (1/2) * Tridiagonal(ones(N-1), zeros(N), ones(N-1))
    Q[1,2] = 1
    Q[N,N-1] = 1
    μ = zeros(N)
    ν = zeros(N)
    μ[1] = N + 1
    ν[N] = N + 1
    γ,d = BBD(Q, μ, ν, n, verbose=true)
    return (γ, d)
end

function plot_midpoint(N, n)
    c, d = gromov_convergence(N, n)
    ρ = c.vector.ρ
    f = Figure()
    ax = Axis(f[1,1])
    lines!(ax, collect(1:N), ρ[n ÷ 2,:])
    current_figure()

end
