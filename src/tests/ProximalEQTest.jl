include("Inclusion.jl")
include("../galerkin/ProximalEqualityIndicator.jl")

"""
    test_proj_IJeq()

Description of the function.

#TODO
"""
function test_proj_IJeq()
    """
    this test projects ρ and q onto the set Jeq by computing their average
    """
    ρ = [1/3; 1/3; 1/3;; 2/3; 1/6; 1/6;; 1; 0; 0]
    q = [1/2; 1/2; 0;; 3/4; 1/4; 0;; 1; 0; 0]
    answer = [5/12; 5/12; 1/6;; 17/24; 5/24; 1/12;; 1; 0; 0]
    ρ_pr, q_pr = project_IJeq(ρ, q)
    ec = 0
    try
	    @assert isapprox(answer, ρ_pr)
    catch e
        println(e)
        ec += 1
    end
    try
	    @assert isapprox(answer, q_pr)
    catch e
        println(e)
        ec += 1
    end
    try
	    @assert is_in_JEq(ρ_pr, q_pr)
    catch e
        println(e)
        ec += 1
    end
    return ec
end
