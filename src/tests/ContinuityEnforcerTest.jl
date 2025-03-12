include("../galerkin/ContinuityEnforcer.jl")
include("Inclusion.jl")

function test_system_formation()
    """
    this test forms the linear system for projecting onto the set of solutions to the Galerkin-discretized discrete continuity
    eqaution. A was computed by hand for a simple graph on 2 nodes, with 6 time steps.
    """
    N = 6
    Q = [0 1; 1 0]
    system = form_ceh_system(Q, N)
    A =
           [-37 1  36 0 0 0 0 0 0 0 0 0 1;
             1 -37 0 36 0 0 0 0 0 0 0 0 1;

            36 0 -73   1 36 0 0 0 0 0 0 0 1;
            0 36   1 -73 0 36 0 0 0 0 0 0 1;

            0 0 36 0 -73 1 36 0 0 0 0 0 1;
            0 0 0 36 1 -73 0 36 0 0 0 0 1;

            0 0 0 0 36 0 -73 1 36 0 0 0 1;
            0 0 0 0 0 36 1 -73 0 36 0 0 1;

            0 0 0 0 0 0 36 0 -73 1 36 0 1;
            0 0 0 0 0 0 0 36 1 -73 0 36 1;

            0 0 0 0 0 0 0 0 36 0 -37 1 1;
            0 0 0 0 0 0 0 0 0 36 1 -37 1;
            1 1 1 1 1 1 1 1 1 1  1  1  0 ]
    ec = 0
    try
	    @assert isapprox(A, system)
    catch e
        println(e)
        println(A - system)
        ec += 1
    end
    return ec
end

function test_target_formation()
    """
    this test forms the target vector b for projecting onto the set of solutions to the Galerkin-discretized discrete continuity
    eqaution. the true answer, target, was computed by hand for a simple graph on 2 nodes, with 4 time steps.
    """
    ρ_A = [1; 0]
    ρ_B = [0; 1]
    Q = [0 1; 1 0]
    ρ = [1 0; 0.75 0.25; 0.25 0.75; 0 1]
    m = permutedims(cat([0 0.5; -0.5 0],[0 0.5; -0.5 0], [0 0.5; -0.5 0], dims=3), (3, 1, 2))
    v = form_b(ρ_A, ρ_B, ρ, m, Q)
    target = -1 * [-5/4, 5/4, -2, 2, -5/4, 5/4, 0]
    ec = 0
    try
	    @assert isapprox(v, target)
    catch e
        println(e)
        println(target)
        println(v)
        ec += 1
    end
    return ec
end

function test_enforcer()
    """
    this test projects a pair(ρ, m) onto the set of solutions to the Galerkin-discretized discrete continuity
    eqaution. the true solution vector, φ, was computed in Mathmetica
    """
    ρ_A = [1; 0]
    ρ_B = [0; 1]
    Q = [0 1; 1 0]
    v = [0.5; 0.5]
    ρ = [1 0; 0.75 0.25; 0.25 0.75; 0 1]
    m = permutedims(cat([0 0.5; -0.5 0],[0 0.5; -0.5 0], [0 0.5; -0.5 0], dims=3), (3, 1, 2))
    φ = (1 / 116) * [-86 86; -89 89; -86 86]
    ∇φ = [0 0; -3/116 3/116; 3/116 -3/116; 0 0]
    ρ_pr = ρ + 3 * ∇φ
    m_pr = m + permutedims(cat([0 -86/58; 86/58 0], [0 -89/58; 89/58 0], [0 -86/58; 86/58 0], dims=3), (3,1,2))
    ρ_hat, m_hat = proj_CE(ρ, m, ρ_A, ρ_B, Q)
    ec = 0
    try
	    @assert isapprox(ρ_hat, ρ_pr)
    catch e
        println(e)
        ec += 1
    end
    try
	    @assert isapprox(m_hat, m_pr)
    catch e
        println(e)
        ec += 1
    end

	is_in_CE(ρ_pr, m_pr, Q, v)
    #try
	    #@assert is_in_CE(ρ_pr, m_pr, Q, v)
    #catch err
        #println(err)
        #ec +=1
    #end

    proj_CE!(ρ, m, ρ_A, ρ_B, Q)

    try
	    @assert isapprox(ρ, ρ_pr)
    catch e
        println(ρ .- ρ_pr)
        println(e)
        ec += 1
    end
    try
	    @assert isapprox(m, m_pr)
    catch e
        println(e)
        println(m .- m_pr)
        ec += 1
    end

    is_in_CE(ρ, m, Q, v)
    #try
	    #@assert is_in_CE(ρ, m, Q, v)
    #catch err
        #println(err)
        #ec +=1
    #end


    return ec
end

println(test_target_formation() + test_system_formation() + test_enforcer())
