include("ChambolleTest.jl")

"""
    step!(a, b, a_bar, a_next, b_next, a_bar_next, c, d; σ = 0.5, τ = 0.5, λ = 1.0)

Description of the function.

#TODO
"""
function step!(a, b, a_bar, a_next, b_next, a_bar_next, c, d; σ = 0.5, τ = 0.5, λ = 1.0)

    combine!(c, b, a_bar, 1.0, σ)
    prox_Fstar!(b_next, c)
    combine!(d, a, b_next, 1.0, -τ)
    prox_G!(a_next, d)
    combine!(d, a_next, a, 1.0, -1.0)
    combine!(a_bar_next, a_next, d, 1.0, λ)
end

function update!(a, b, a_bar, a_next, b_next, a_bar_next, c, d; σ = 0.5, τ = 0.5, λ = 1.0)
    assign!(a, a_next)
    assign!(b, b_next)
    assign!(a_bar, a_bar_next)
end

"""
    export_stepper_seed()

Description of the function.

#TODO
"""
function export_stepper_seed()
    Q = [
        0.  0.5 0.  0.5;
        0.5 0.  0.5 0.;
        0.  0.5 0.  0.5;
        0.5 0.  0.5 0.;
    ]
    #μ = [3.; 1/3; 1/3; 1/3]
    #ν = [1/3; 3.; 1/3; 1/3]
    μ = [4.; 0.; 0.; 0.;]
    ν = [0.; 4.; 0.; 0.;]
    N = 3

    a = ErbarBundle(Q, μ, ν, N)
    b = ErbarBundle(Q, μ, ν, N)
    a_bar = ErbarBundle(Q, μ, ν, N)
    b_next = ErbarBundle(Q, μ, ν, N)
    a_next = ErbarBundle(Q, μ, ν, N)
    a_bar_next = ErbarBundle(Q, μ, ν, N)
    c = ErbarBundle(Q, μ, ν, N)
    d = ErbarBundle(Q, μ, ν, N)
    σ = 0.5
    τ = 0.5

    return (a, b, a_bar, a_next, b_next, a_bar_next, c, d)
end

#a_next, b_next, a_bar_next, c, σ, τ = chambolle_pock_routine_stepper(a, b, a_bar, σ, τ)
