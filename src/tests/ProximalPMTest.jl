include("Inclusion.jl")
include("../galerkin/ProximalSignIndicator.jl")
using CUDA

function test()
    """
    this test computes the proximal mapping of IJpm_star at (ρ, ρ_minus, ρ_plus)
    via Moreau's identity. The projections were computed by hand.
    """
	Q = [0.; 1.;; 1.; 0]
    ρ = [1.; 0.;; 0.; 1]
    ρ_minus = [1.; 4.;; 2.; 5.;;; 2.; 5.;; 3.; 6]
    ρ_plus = [7.; 10.;; 8.; 11.;;; 8.; 11.;; 9.; 12]
    ρ_hat, ρ_minus_hat, ρ_plus_hat = proximal_IJpm_star(ρ, ρ_minus, ρ_plus, Q)

    # hand computed projections
    ρ_proj = [3; 4;; 5/2; 9/2]
    ρ_plus_proj = [3; 4;; 3; 4;;; 5/2; 9/2;; 5/2; 9/2]
    ρ_minus_proj = [3; 4;; 5/2; 9/2;;; 3; 4;; 5/2; 9/2]
    ec = 0
    try
        @assert ρ_hat == (ρ .- ρ_proj)
    catch e
        println(e)
        println(ρ_hat)
        println(ρ_proj)
        ec += 1
    end
    try
        @assert ρ_minus_hat == (ρ_minus .- ρ_minus_proj)
    catch e
        println(e)
        ec += 1
    end
    try
        @assert ρ_plus_hat == (ρ_plus .- ρ_plus_proj)
    catch e
        println(e)
        ec += 1
    end
    try
	    @assert is_in_J_PM(ρ .- ρ_hat, ρ_minus .- ρ_minus_hat, ρ_plus .- ρ_plus_hat)
    catch e
        println(e)
        ec += 1
    end

    # if you here, you passed the none mutating function tests

    proximal_IJpm_star!(ρ, ρ_minus, ρ_plus, Q)

    try
        @assert ρ_hat == ρ
    catch e
        println(e)
        ec += 1
    end
    try
        @assert ρ_minus_hat == ρ_minus
    catch e
        println(e)
        ec += 1
    end
    try
        @assert ρ_plus_hat == ρ_plus
    catch e
        println(e)
        ec += 1
    end

    return ec
end

function test_gpu()
    """
    this test computes the proximal mapping of IJpm_star at (ρ, ρ_minus, ρ_plus)
    via Moreau's identity, but now we work with CUDA arrays. The projections were computed by hand.
    """
    # constants
	Q = CuArray([0.; 1.;; 1.; 0])
    ρ = CuArray([1.; 0.;; 0.; 1])
    ρ_minus = CuArray([1.; 4.;; 2.; 5.;;; 2.; 5.;; 3.; 6])
    ρ_plus = CuArray([7.; 10.;; 8.; 11.;;; 8.; 11.;; 9.; 12])
    ρ_proj = CuArray([3; 4;; 5/2; 9/2])
    ρ_plus_proj = CuArray([3; 4;; 3; 4;;; 5/2; 9/2;; 5/2; 9/2])
    ρ_minus_proj = CuArray([3; 4;; 5/2; 9/2;;; 3; 4;; 5/2; 9/2])

    ρ_hat = ρ .- ρ_proj
    ρ_minus_hat = ρ_minus .- ρ_minus_proj
    ρ_plus_hat = ρ_plus .- ρ_plus_proj

    #compute answers
    proximal_IJpm!(ρ, ρ_minus, ρ_plus, Q)

    try
        @assert ρ ≈ ρ_hat
    catch e
        println(e)
        println(ρ_hat)
        println(ρ_proj)
        ec += 1
    end
    try
        @assert ρ_minus_hat ≈ ρ_minus
    catch e
        println(e)
        ec += 1
    end
    try
        @assert ρ_plus_hat ≈ ρ_plus
    catch e
        println(e)
        ec += 1
    end


end

