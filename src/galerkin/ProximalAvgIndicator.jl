include("../utils.jl")
using SparseArrays
using LinearAlgebra

"""
this file contains the functions needed to compute the proximal mapping of IJ_avg_star
as described in Erbar et al 2020, section 4.6.

Let h be the step size, let n be the number of nodes, we are given
ρ ∈ V_{n × h}^1 (i.e. a matrix of size (1 + 1/h) × n)
ρ_avg ∈ V_{n × h}^0 (i.e. a matrix of size 1/h × n)

IJ_avg is an indicator function, so it's proximal mapping is a projection. Thus Moreau's identity
prox_f_star = identity - prox_f allows us to compute the proximal mapping of IJ_avg_star
by solving a projection problem and subtracting the answer from the argument. Thus one computes

prox_IJavg_star(ρ,ρ_avg) = (ρ, ρ_avg) - proj_Javg(ρ, ρ_avg)

A Lagrange multipliers argument leads to the conclusion that the solution to the projection problem
is given in terms of the solution of a linear system.

"""

# only gets called once, when initializing a vector
function form_avg_system(N)
    """
    the projection problem is equivalent to solving a linear system Mx=b, whose dimension depends upon the
    stepsize N ≡ 1/h
    the matrix is always the same, so the function should only ever be called when initializing an ErbarCache
    """
    off_diag = ones(N - 1)
    diag = 6 * ones(N)
    diag[1] = 5
    diag[N] = 5
    M = Tridiagonal(off_diag, diag, off_diag)
    return lu(sparse(0.25 * M))
end

# functions needed to solve the relevant problem
function form_b(ρ, ρ_bar, ρ_A, ρ_B)
    """
    this function computes the vectors b in the aforementioned linear system. One has, as per
    the last (unnumbered) eqn in Erbar Section 4.6

    b = [
         ρ_bar(t_0, x) - (1/2)*(ρ_A + ρ(t_1,x))
         ...,
         ρ_bar(t_i, x) - (1/2)*(ρ(t_{i+1}) + ρ(t_{i}, x))
         ...,
         ρ_bar(t_{N-1}, x) - (1/2)*(ρ_B + ρ(t_{N-1}, x))
    ]

    all the vectors b are computed simultaneously and returned in the matrix B
    """

    N = size(ρ, 1)
    B = copy(ρ_bar)
    B[1,:] .-= 0.5 * (ρ_A + ρ[2,:])
    B[2:N-2,:] .-= 0.5 * (ρ[3:N-1,:] + ρ[2:N-2,:])
    B[N-1,:] .-= 0.5 * (ρ_B + ρ[N - 1,:])

    return B
end

function prox_IJavg_star(ρ, ρ_bar, ρ_A, ρ_B, M)
    """
    compute the proximal mapping of IJavg_star via Moreau's identity

    arguments
    Let h be the step size, let n be the number of nodes, we are given
    ρ ∈ V_{n × h}^1 (i.e. a matrix of size (1 + 1/h) × n)
    ρ_avg ∈ V_{n × h}^0 (i.e. a matrix of size 1/h × n)
    ρ_A, ρ_B ∈ R^n are column vectors
    M is the matrix defining the linear system to be solved
    """
    N, V = size(ρ)
    Λ = similar(ρ_bar)
    B = form_b(ρ, ρ_bar, ρ_A, ρ_B)
	Λ = M \ B

    ρ_bar_pr = ρ_bar - Λ
    Λ_bar = vcat(zeros(V)', 0.5 * (Λ[2:N-1,:] + Λ[1:N-2,:]), zeros(V)')
    ρ_pr = ρ + Λ_bar
    ρ_pr[1,:] = ρ_A
    ρ_pr[N,:] = ρ_B

    return (ρ - ρ_pr, ρ_bar - ρ_bar_pr)

end

function prox_IJavg_star!(ρ, ρ_bar, ρ_A, ρ_B, M)
    """
    compute inplace the proximal mapping of IJavg_star via Moreau's identity

    arguments
    Let h be the step size, let n be the number of nodes, we are given
    ρ ∈ V_{n × h}^1 (i.e. a matrix of size (1 + 1/h) × n)
    ρ_avg ∈ V_{n × h}^0 (i.e. a matrix of size 1/h × n)
    ρ_A, ρ_B ∈ R^n are column vectors
    M is the matrix defining the linear system to be solved
    """
    N, V = size(ρ)
    B = form_b(ρ, ρ_bar, ρ_A, ρ_B)
	Λ = M \ B

    ρ[1,:] .-= ρ_A
    ρ[2:N-1,:] .= -0.5 * (Λ[1:N-2,:] + Λ[2:N-1,:])
    ρ[N,:] .-= ρ_B

    ρ_bar .= Λ

end
