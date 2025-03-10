using BenchmarkTools
using CUDA
include("../galerkin/ProximalAction.jl")

function test()
    """
    this test works be generating a random point on the parabola, moving out orthogonally
    by a randome value λ, and then projecting back.
    """
    γ(t) = [-0.25 * t^2; t]
    N(t) = [1, 0.5 * t]
    s = rand()
    λ = rand()
    p = γ(s)
    v = p .+ (λ * N(s))
    x, y = proj_B(v[1], v[2])
    try
        @assert isapprox(x, p[1], atol=1e-8) && isapprox(y, p[2], atol=1e-8)
    catch e
        println(e)
        println(p)
        println([x, y])
        println(v)
    end
end

function test_gpu()
    """
    similar to the above test, but now with CUDA arrays
    """
    γ(t) = [-0.25 * t^2; t]
    N(t) = [1; 0.5 * t]

    T = zeros(10,10,10)
    S = zeros(10,10,10)
    Tproj = zeros(10,10,10)
    Sproj = zeros(10,10,10)

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

println(test())
