using LinearAlgebra
using CairoMakie
include("EarthMover.jl")
include("utils.jl")



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

## Two Points
function diracs_on_two_points(;N=128, ε=0., verbose=false)
    Q = [0. 1.; 1. 0.]
    μ = [2.; 0]
    ν = [0.; 2]
    a = [-1; 1;]
    b = [1; -1;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N, verbose=verbose)
    return γ, d
end


## Triangle

function diracs_on_triangle(;N = 128, ε = 0., verbose=false)
    Q = [0. 0.5 0.5; 0.5 0. 0.5; 0.5 0.5 0.]
    μ = [3.; 0; 0.]
    ν = [0.; 3; 0.]
    a = [-1; 1/2; 1/2;]
    b = [1/2; -1; 1/2;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N, verbose=verbose)
    return (γ, d)
end

## Square
function diracs_on_square(;N=128, ε=0., verbose=false)
    Q = [0. 0.5 0. 0.5;
         0.5 0. 0.5 0.;
         0. 0.5 0. 0.5;
         0.5 0. 0.5 0.]
    μ = [4.; 0; 0.; 0.]
    ν = [0.; 4; 0.; 0.]
    a = [-1; 1/3; 1/3; 1/3;]
    b = [1/3; -1; 1/3; 1/3;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N, verbose=verbose)
    return (γ, d)
end
## 9x9 Grid

function diracs_on_grid(; N=128, ε=0., verbose=false)
    Q = Array(Tridiagonal(ones(7), zeros(8), ones(7)))
    Q[1, 8] = 1
    Q[8, 1] = 1
    Q = hcat(Q, zeros(8))
    Q = vcat(Q, zeros(9)')
    Q[9,2] = Q[2,9] = Q[9,4] = Q[4,9] = Q[9,6] = Q[6,9] = Q[9,8] = Q[8,9] = 1

    for (idx, row) in enumerate(eachrow(Q))
        z = sum(row)
        Q[idx, :] /= z
    end

    π = steady_state_from_adjacency(Q)
    μ = zeros(9)
    ν = zeros(9)
    μ[3] = 1/π[3]
    ν[7] = 1/π[7]

    γ, d = BBD(Q, μ, ν, N, verbose=verbose)
    return (γ, d)

end
