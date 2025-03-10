include("Inclusion.jl")
include("../galerkin/ProximalAvgIndicator.jl")
using CUDA

function test()
    """
    this test computes the proximal mapping of IJAvg at (ρ, ρ_bar). The true basis for the true solution Λ
    was computed externally.
    """
    N = 3
	ρ_A = [1.; 0.; 0.]
    ρ_B = [0.; 0.; 1.]
    ρ = [
        1 0 0;
        1/2 1/2 0;
        0 1/2 1/2;
        0 0 1
    ]

    ρ_bar = [
        1. 0. 0.;
        0. 1. 0.;
        0. 0. 1.;
    ]
    Λ = [
        17/70 -2/7 3/70;
        -3/14 3/7 -3/14;
        3/70 -2/7 17/70;
    ]
    ρ_pr = [
        0 0 0;
        -1/70 -5/70 6/70;
        6/70 -5/70 -1/70;
        0 0 0;
    ]

    ρ_pr_computed, ρ_bar_pr_computed = prox_IJavg_star(ρ, ρ_bar, ρ_A, ρ_B, form_avg_system(N))

    ec = 0

    try
	    @assert ρ_pr_computed ≈ ρ_pr
    catch e
        println(e)
        ec += 1
    end

    try
	    @assert ρ_bar_pr_computed ≈ Λ
    catch e
        println(e)
        ec += 1
    end
    # if you here, you passed the none mutating function tests

    prox_IJavg_star!(ρ, ρ_bar, ρ_A, ρ_B, form_avg_system(N))

    try
        @assert ρ_pr_computed ≈ ρ
    catch e
        println(e)
        ec += 1
    end
    try
        @assert ρ_bar ≈ Λ
    catch e
        println(ρ_bar .- Λ)
        println(e)
        ec += 1
    end
    return ec

end

function test_gpu()
    #constants
    N = 3
	ρ_A = CuArray([1.; 0.; 0.])
    ρ_B = CuArray([0.; 0.; 1.])
    ρ = CuArray([
        1 0 0;
        1/2 1/2 0;
        0 1/2 1/2;
        0 0 1
    ])

    ρ_bar = CuArray([
        1. 0. 0.;
        0. 1. 0.;
        0. 0. 1.;
    ])

    # true answers
    Λ = CuArray([
        17/70 -2/7 3/70;
        -3/14 3/7 -3/14;
        3/70 -2/7 17/70;
    ])

    ρ_pr = CuArray([
        0 0 0;
        -1/70 -5/70 6/70;
        6/70 -5/70 -1/70;
        0 0 0;
    ])

    prox_IJavg!(ρ, ρ_bar, ρ_A, ρ_B, CuArray(form_avg_system(N)))

    ec = 0

    try
	    @assert ρ_pr ≈ ρ
    catch e
        println(e)
        println(ρ_pr)
        println(ρ)
        ec += 1
    end

    try
	    @assert ρ_bar ≈ Λ
    catch e
        println(e)
        ec += 1
    end

end
