include("../ErbarVector.jl")
include("../galerkin/Chambolle.jl")


Q = [0 1; 1 0]
μ = [2; 0]
ν = [0; 2]
N = 3

σ = 0.5
τ = 0.5
λ = 1.0

function chambolle_pock_routine_stepper(
    Q::AbstractMatrix,
    μ::AbstractVector,
    ν::AbstractVector,
    N::Int64;
    maxiters=1000,
    σ=0.5,
    τ=0.5,
    λ=1.0,
    tol=1e-3
)
    a = ErbarBundle(Q, μ, ν, N)
    b = ErbarBundle(Q, μ, ν, N)
    a_bar= ErbarBundle(Q, μ, ν, N)
    mutable_a = ErbarBundle(Q, μ, ν, N)
    mutable_b = ErbarBundle(Q, μ, ν, N)
    mutable_a_bar= ErbarBundle(Q, μ, ν, N)
    mutable_a_next = ErbarBundle(Q, μ, ν, N)
    mutable_b_next = ErbarBundle(Q, μ, ν, N)
    mutable_a_bar_next = ErbarBundle(Q, μ, ν, N)
    mutable_c = ErbarBundle(Q, μ, ν, N)
    mutable_d = ErbarBundle(Q, μ, ν, N)
    a_comp = ErbarBundle(Q, μ, ν, N)
    b_comp = ErbarBundle(Q, μ, ν, N)
    a_bar_comp = ErbarBundle(Q, μ, ν, N)
    d_comp = ErbarBundle(Q, μ, ν, N)


    for i in 1:maxiters
        println("ITER $i")
        #form argument c for Fstar, c = b + σ*a_bar
        combine!(mutable_c, mutable_b, mutable_a_bar, 1.0, σ)
        # b_next = F_star(c = b + σ * a_bar)
        prox_Fstar!(mutable_b_next, mutable_c)
        #form argument c for G, c = a - τ * b_next
        combine!(mutable_c, mutable_a, mutable_b_next, 1.0, -τ)
        prox_G!(mutable_a_next, mutable_c)
        combine!(mutable_d, mutable_a_next, mutable_a, 1.0, -1.0)
        combine!(mutable_a_bar_next, mutable_a_next, mutable_d, 1.0, λ)
        assign!(mutable_a, mutable_a_next)
        assign!(mutable_b, mutable_b_next)
        assign!(mutable_a_bar, mutable_a_bar_next)

        b_next = prox_Fstar(σ, b, a_bar)
        a_next = prox_G(τ, a, b_next)
        d = a_next - a
        a_bar_next = a_next + λ * d
        a = a_next
        b = b_next
        a_bar = a_bar_next

        combine!(a_comp, a, mutable_a, 1.0, -1.0)
        combine!(b_comp, b, mutable_b, 1.0, -1.0)
        combine!(a_bar_comp, a_bar, mutable_a_bar, 1.0, -1.0)
        combine!(d_comp, d, mutable_d, 1.0, -1.0)
        println("MUTABLE VS IMMUTABLE NORM DIFFERENCES")
        println(norm(a_comp))
        println(norm(b_comp))
        println(norm(a_bar_comp))
        println(norm(d_comp))
        v = d.vector
        u = mutable_d.vector

        #println("************ρ")
        #println(v.ρ - u.ρ)
        #println("************θ")
        #println(v.θ - u.θ)
        #println("************m")
        #println(v.m - u.m)
        #println("************ρ_minus")
        #println(v.ρ_minus - u.ρ_minus)
        #println("************ρ_plus")
        #println(v.ρ_plus - u.ρ_plus)
        #println("************ρ_avg")
        #println(v.ρ_avg - u.ρ_avg)
        #println("************q")
        #println(v.q - u.q)
#
        #if i == maxiters
            #return (u, v)
        #end

    end

end

chambolle_pock_routine_stepper(Q, μ, ν, N)

#Abar = ErbarBundle(Q, μ, ν, N)
#A = ErbarBundle(Q, μ, ν, N)
#B = ErbarBundle(Q, μ, ν, N)
#C = ErbarBundle(Q, μ, ν, N)
#D = ErbarBundle(Q, μ, ν, N)
#
#Abarhat = ErbarBundle(Q, μ, ν, N)
#Ahat = ErbarBundle(Q, μ, ν, N)
#Bhat = ErbarBundle(Q, μ, ν, N)
#Dhat = ErbarBundle(Q, μ, ν, N)
#
#combine!(C, B, Abar, 1.0, σ)
#Bhat = prox_Fstar(σ, B, Abar)
#prox_Fstar!(B, C)
#
#combine!(D, B, Bhat, 1.0, -1.0)
#println(norm(D))
#
#println("*********PROX G")
#combine!(C, B, Abar, 1.0, -σ)
#Bhat = prox_G(σ, B, Abar)
#prox_G!(B, C)
#
#combine!(D, B, Bhat, 1.0, -1.0)
#println(norm(D))
#
