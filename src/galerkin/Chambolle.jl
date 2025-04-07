include("../ErbarVector.jl")
include("./ContinuityEnforcer.jl")
include("./ProximalAvgIndicator.jl")
include("./ProximalEqualityIndicator.jl")
include("./ProximalSignIndicator.jl")
include("./ProximalAction.jl")
include("./KProjection.jl")
using ProgressMeter


"""
this file contains the Chambolle pock routine for minimizing the objective function described in eqn (26)
of Erbar et al 2020.

RELEVANT SETS:

Javg = { ρ, ρ_bar : ρ_bar == avg_h(ρ) } (avg_h is the time average operator, see definition 3.2 in Erbar 2020)
Jeq = { ρ_bar, q : ρ_bar == q }
Jpm = { q, ρ_minus, ρ_plus : ρ_minus[t,i,j] == q[t,i] && ρ_plus[t,i,j] == q[t,j]}
K = { x,y,z : 0 ≤ z ≤ T(x,y) } (here T is the chosen admissible mean)
ScriptK = { ρ_minus, ρ_plus, θ: ρ_minus[t,i,j], ρ_plus[t,i,j], θ[t,i,j] ∈ K}
"""

# a more memory efficient version of ChamPock
function chambolle_pock_me(
    Q::AbstractMatrix,
    μ::AbstractVector,
    ν::AbstractVector,
    N::Int;
    maxiters=2^16,
    σ=0.5,
    τ=0.5,
    λ=1.0,
    tol=1e-10,
    gpu=false
)
    """
    this is a memory efficient version of Chambolle Pock that does computations in place whenever possible
    iterations cease if ∫||ρ_{k} - ρ_{k + 1}||_π dt < tol

    arguments
    Q, a Markov kernel defining the graph
    μ, ν, probability densities w.r.t to the steady state π of Q
    N, the number of steps in the geodesic
    """
    # we will only ever use 8 vectors
    a = ErbarBundle(Q, μ, ν, N, gpu=gpu)
    b = ErbarBundle(Q, μ, ν, N, gpu=gpu)
    a_bar = ErbarBundle(Q, μ, ν, N, gpu=gpu)
    a_next = ErbarBundle(Q, μ, ν, N, gpu=gpu)
    b_next = ErbarBundle(Q, μ, ν, N, gpu=gpu)
    a_bar_next = ErbarBundle(Q, μ, ν, N, gpu=gpu)
    c = ErbarBundle(Q, μ, ν, N, gpu=gpu)
    d = ErbarBundle(Q, μ, ν, N, gpu=gpu)
    p = ProgressUnknown(spinner=true)


    function assert_unique_pointers(s,t,u,v,w,x,y,z)
        ptrs = []
	    for bundle in [s,t,u,v,w,x,y,z]
            v = bundle.vector
            for arr in [v.ρ, v.m, v.θ, v.ρ_minus, v.ρ_plus, v.ρ_avg, v.q]
                push!(ptrs, pointer(arr))
            end
        end
        @assert allunique(ptrs)
    end



    for i in 1:maxiters
        next!(p)

        combine!(c, b, a_bar, 1.0, σ)
        prox_Fstar!(b_next, c)
        combine!(c, a, b_next, 1.0, -τ)
        prox_G!(a_next, c)
        try
            @assert !any(a_next.vector.ρ .< 0)
        catch error
            println(i)
            println("Negative mass found after CE projection in prox_G call")
            println(minimum(a_next.vector.ρ))
            return a
            #error("Nonnegative mass!")
        end


        combine!(d, a_next, a, 1.0, -1.0)
        normdiff = sum(d.vector.ρ .* d.vector.ρ * d.cache.π)
        if normdiff < tol
            println("converged on iter $i")
            return a
        end
        λ = 1 / √(1 + 2 * τ)
        τ *= λ
        σ /= λ

        combine!(a_bar_next, a_next, d, 1.0, λ)
        assign!(a, a_next)
        assign!(b, b_next)
        assign!(a_bar, a_bar_next)
        #assert_unique_pointers(a,b,a_bar,a_next,b_next, a_bar_next, c, d)
    end
    error("Chambolle Pock did not converge in $(maxiters) steps")
    #return a
end

function chambolle_pock_routine(
    Q::Matrix{AbstractFloat},
    μ::Vector{AbstractFloat},
    ν::Vector{AbstractFloat},
    N::Int64;
    maxiters=1000,
    σ=0.5,
    τ=0.5,
    λ=1.0,
    tol=1e-3
)
    """
    the Chambolle Pock routine. vectors are reallocated on every iteration.
    iterations cease if ∫||ρ_{k} - ρ_{k + 1}||_π dt < tol

    arguments
    Q, a Markov kernel defining the graph
    μ, ν, probability densities w.r.t to the steady state π of Q
    N, the number of steps in the geodesic
    """
    a = ErbarBundle(Q, μ, ν, N)
    b = ErbarBundle(Q, μ, ν, N)
    a_bar= ErbarBundle(Q, μ, ν, N)

    for _ in 1:maxiters
        b_next = prox_Fstar(σ, b, a_bar)
        a_next = prox_G(τ, a, b_next)
        d = a_next - a
        a_bar_next = a_next + λ * d
        normdiff = sum(d.vector.ρ .* d.vector.ρ * d.cache.π)
        if normdiff < tol
            return a_next
        end
        a = a_next
        b = b_next
        a_bar = a_bar_next
    end

    return a

end


function prox_Fstar!(targ, bundle)
    """
    compute the proximal mapping of F star in place
    this amounts to computing the proximal mappings of the conjuage Action, IJPM, and IJAvg
    """
    cache = bundle.cache
    v = bundle.vector
    u = targ.vector
    prox_Astar!(v.θ, v.m)
    proximal_IJpm_star!(v.q, v.ρ_minus, v.ρ_plus, cache.Q)
    prox_IJavg_star!(v.ρ, v.ρ_avg, cache.μ, cache.ν, cache.avg_sys)
    u.ρ .= v.ρ
    u.θ .= v.θ
    u.m .= v.m
    u.ρ_minus .= v.ρ_minus
    u.ρ_plus .= v.ρ_plus
    u.ρ_avg .= v.ρ_avg
    u.q .= v.q
end

function prox_G!(targ, bundle)
    """
    compute the proximal mapping of G
    this amounts to computing the projection to the space of solutions to the Galerkin-discretized continuity equation,
    projection to the set Script K, and projection to the set Jeq
    """
    cache = bundle.cache
    v = bundle.vector
    u = targ.vector
    proj_CE!(v.ρ, v.m, cache.μ, cache.ν, cache.Q, cache.ceh_sys)
    project_K!(v.ρ_minus, v.ρ_plus, v.θ)
    project_IJeq!(v.ρ_avg, v.q)

    u.ρ .= v.ρ
    u.θ .= v.θ
    u.m .= v.m
    u.ρ_minus .= v.ρ_minus
    u.ρ_plus .= v.ρ_plus
    u.ρ_avg .= v.ρ_avg
    u.q .= v.q
end


function prox_Fstar(σ::AbstractFloat, b::ErbarBundle, a_bar::ErbarBundle)
    """
    compute the proximal mapping of F star
    this amounts to computing the proximal mappings of the conjuage Action, IJPM, and IJAvg
    """
    cache = b.cache
    u = b + σ * a_bar
    v = u.vector
    θ, m = prox_Astar(v.θ, v.m)
    q, ρ_minus, ρ_plus = proximal_IJpm_star(v.q, v.ρ_minus, v.ρ_plus, cache.Q)
    ρ, ρ_avg = prox_IJavg_star(v.ρ, v.ρ_avg, cache.μ, cache.ν, cache.avg_sys)
    vprime = ErbarVector(ρ, m, θ, ρ_minus, ρ_plus, ρ_avg, q)
    return ErbarBundle(b.cache, vprime)

end

function prox_G(τ::AbstractFloat, a::ErbarBundle, b::ErbarBundle)
    """
    compute the proximal mapping of G
    this amounts to computing the projection to the space of solutions to the Galerkin-discretized continuity equation,
    projection to the set Script K, and projection to the set Jeq
    """
    cache = a.cache
    u = a - (τ * b)
    v = u.vector

    ρ, m = proj_CE(v.ρ, v.m, cache.μ, cache.ν, cache.Q, cache.D, cache.ceh_sys)
    ρ_minus, ρ_plus, θ = project_K(v.ρ_minus, v.ρ_plus, v.θ)
    ρ_avg, q = project_IJeq(v.ρ_avg, v.q)

    vprime = ErbarVector(ρ, m, θ, ρ_minus, ρ_plus, ρ_avg, q)
    return ErbarBundle(a.cache, vprime)
end
