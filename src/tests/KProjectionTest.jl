include("../galerkin/KProjection.jl")


"""
    test_proj_ktop()

Description of the function.

#TODO
"""
function test_proj_ktop()
    """
    this test works be generating a random point on the surface K, moving out orthogonally
    by a randome value λ, and then projecting back.
    """
    γ(s,t) = [s; t; √(s*t)]
    N(s,t) = -0.5 * [√(t/s); √(s/t); -2.]
    t, s = rand(), rand()
    λ = rand()
    p = γ(s,t)
    v = p .+ (λ * N(s,t))
    x, y, z = project_by_claude(v[1], v[2], v[3])
    try
        @assert isapprox(x, p[1], atol=1e-10)
        @assert isapprox(y, p[2], atol=1e-10)
        @assert isapprox(z, p[3], atol=1e-10)
    catch e
        println(e)
        println(p)
        println([x, y, z])
        println(v)
    end
end

