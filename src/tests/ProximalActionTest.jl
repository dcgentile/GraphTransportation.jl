using BenchmarkTools
using CUDA
include("../galerkin/ProximalAction.jl")

"""
    test_proj_B(n)

Description of the function.

#TODO
"""
function test_proj_B(n)
    """
    this test works be generating a random point on the parabola, moving out orthogonally
    by a randome value λ, and then projecting back.
    """
    γ(t) = [-0.25 * t^2; t]
    N(t) = [1, 0.5 * t]
    s = n*rand()
    λ = n*rand()
    p = γ(s)
    v = p .+ (λ * N(s))
    #x, y = proj_B(v[1], v[2])
    x, y = projection_by_newton(v[1], v[2])
    try
        @assert isapprox(x, p[1], atol=1e-8) && isapprox(y, p[2], atol=1e-8)
    catch e
        println(e)
        println(p)
        println([x, y])
        println(v)
    end
    println("Successfully projected to the parabola x + y^2 / 4 = 0")
end

"""
    test_action()

Description of the function.

#TODO
"""
function test_action()
    """
    similar to the above test, but now with CUDA arrays
    """
    γ(t) = [-0.25 * t^2; t]
    N(t) = [1; 0.5 * t]

    M = 2
    T = zeros(M,M,M)
    S = zeros(M,M,M)
    Tproj = zeros(M,M,M)
    Sproj = zeros(M,M,M)

    for i in eachindex(T)
        s = rand()
        p = γ(s)
        λ = rand()
        v = p .+ (λ * N(s))
        Tproj[i] = p[1]
        Sproj[i] = p[2]
        T[i] = v[1]
        S[i] = v[2]
    end

    prox_Astar!(T, S)
    try
        @assert T ≈ Tproj
    catch err
        println(T - Tproj)
    end
    try
        @assert S ≈ Sproj
    catch err
        println(S - Sproj)
    end


end

"""
    test_gpu()

Description of the function.

#TODO
"""
function test_gpu()
    """
    similar to the above test, but now with CUDA arrays
    """
    γ(t) = [-0.25 * t^2; t]
    N(t) = [1; 0.5 * t]

    N = 2

    T = zeros(N,N,N)
    S = zeros(N,N,N)
    Tproj = zeros(N,N,N)
    Sproj = zeros(N,N,N)

    for i in eachindex(T)
        s = rand()
        p = γ(s)
        λ = rand()
        v = p .+ (λ * N(s))
        Tproj[i] = p[1]
        Sproj[i] = p[2]
        T[i] = v[1]
        S[i] = v[2]
    end

    Tproj_gpu = CuArray(Tproj)
    Sproj_gpu = CuArray(Sproj)
    T_gpu = CuArray(T)
    S_gpu = CuArray(S)

    prox_Astar!(T_gpu, S_gpu)
    @assert T_gpu ≈ Tproj_gpu
    @assert S_gpu ≈ Sproj_gpu


end
