include("EarthMover.jl")
using LinearAlgebra

function gromov_convergence(N, n)
    Q = (1/2) * Tridiagonal(ones(N-1), zeros(N), ones(N-1))
    Q[1,2] = 1
    Q[N,N-1] = 1
    μ = zeros(N)
    ν = zeros(N)
    μ[1] = N + 1
    ν[N] = N + 1
    γ,d = BBD(Q, μ, ν, n)
    return (γ, d)

end
