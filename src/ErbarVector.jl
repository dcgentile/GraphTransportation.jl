using SuiteSparse
using SparseArrays
using LinearAlgebra
using CUDA
include("./galerkin/ContinuityEnforcer.jl")
include("./galerkin/ProximalAvgIndicator.jl")


# An ErbarVector is an element of the Hilbert space H, described in eqn (27)
# of Erbar et al. 2020
mutable struct ErbarVector
    # vector components to operate on
    ρ::AbstractArray
    m::AbstractArray
    θ::AbstractArray
    ρ_minus::AbstractArray
    ρ_plus::AbstractArray
    ρ_avg::AbstractArray
    q::AbstractArray
end

# An ErbarCache is an immutable struct that contains the "metadata" of a graph OT
# problem, i.e. those object that we only need to compute and store once
struct ErbarCache
    # cached commonly needed data
    Q::AbstractMatrix            # Markov kernel defining the graph
    μ::AbstractVector            # initial mass configuration
    ν::AbstractVector            # terminal mass configuration
    π::AbstractVector            # steady state distribution of the Markov kernel
    N::Int               # number of steps in geodesic
    ceh_sys      # system defining the Continuity Enforcement problem
    avg_sys      # system defining the Averaging Enforcement problem

    function ErbarCache(Q, μ, ν, N; gpu=false)
        # find steady state of Q
        #λ, v = eigen(Q')
        #_, idx = findmin(abs.(λ .- 1.0))
        #vhat = v[:, idx]
        #π = vhat / sum(vhat)
        V, _ = size(Q)
        S = sparse(Q)
        E = nnz(S)
        #π = ones(V)
        π = zeros(V)
        for i in 1:V
            π[i] = nnz(S[i,:]) / E
        end
        try
            @assert μ' * π ≈ 1
            @assert ν' * π ≈ 1
        catch error
            error("$(μ) or $(ν) is not a density wrt to $(π)")
        end



        # form the linear systems we'll be solving in each step
        ceh_sys = form_ceh_system(Q, N)
        #avg_sys = factorize(form_avg_system(N))
        avg_sys = form_avg_system(N)

        gpu ? new(
            CuArray(Q),
            CuArray(μ),
            CuArray(ν),
            CuArray(π),
            N,
            CuArray(avg_mat),
            CuArray(ceh_sys),
            CuArray(avg_sys)) : new(S, μ, ν, π, N, ceh_sys, avg_sys)

    end
end

# An ErbarBundle is an ErbarCache together with an ErbarVector.
# An EB can be initialized by specifying a Markov kernel Q, two densities w.r.t to
# the kernel's steady state μ, ν, and a number of steps N
mutable struct ErbarBundle
    cache::ErbarCache
    vector::ErbarVector

    function ErbarBundle(cache::ErbarCache, vector::ErbarVector)
	    new(cache, vector)
    end

    function ErbarBundle(Q::AbstractMatrix, μ::AbstractVector, ν::AbstractVector, N::Int; gpu=false)
        cache = ErbarCache(Q, μ, ν, N; gpu=gpu)

        #### INITIALIZE VECTOR COMPONENTS
        V, _ = size(Q)
        ρ = zeros(N + 1, V)
        m = zeros(N, V, V)
        θ = zeros(N, V, V)
        ρ_minus = zeros(N, V, V)
        ρ_plus = zeros(N, V, V)
        ρ_avg = zeros(N,V)
        q = zeros(N,V)

        vector = gpu ? ErbarVector(
            CuArray(ρ),
            CuArray(m),
            CuArray(θ),
            CuArray(ρ_minus),
            CuArray(ρ_plus),
            CuArray(ρ_avg),
            CuArray(q)) : ErbarVector(ρ, m ,θ, ρ_minus, ρ_plus, ρ_avg, q)
        new(cache, vector)

    end
end


Base.copy(a::ErbarCache) = ErbarCache(a.Q, a.μ, a.ν, a.N)
Base.copy(a::ErbarVector) = ErbarVector(copy(a.ρ), copy(a.m), copy(a.θ), copy(a.ρ_minus), copy(a.ρ_plus), copy(a.ρ_avg), copy(a.q))
Base.copy(a::ErbarBundle) = ErbarBundle(copy(a.cache), copy(a.vector))


# combine! computes a linear combination of ErbarVectors and stores the result in a
# pre-allocated vector c
function combine!(c::ErbarVector, a::ErbarVector, b::ErbarVector, α::Number, β::Number)
    c.ρ .= (α .* a.ρ) .+ (β .* b.ρ)
    c.m .= (α .* a.m) .+ (β .* b.m)
    c.θ .= (α .* a.θ) .+ (β .* b.θ)
    c.ρ_minus .= (α .* a.ρ_minus) .+ (β .* b.ρ_minus)
    c.ρ_plus .= (α .* a.ρ_plus) .+ (β .* b.ρ_plus)
    c.ρ_avg .= (α .* a.ρ_avg) .+ (β .* b.ρ_avg)
    c.q .= (α .* a.q) .+ (β .* b.q)
end

# combine! combines the vectors of ErbarBundles a, b, sacled by α,β resp., and stores
# the result in the vector component of c
function combine!(c::ErbarBundle, a::ErbarBundle, b::ErbarBundle, α::Number, β::Number)
    combine!(c.vector, a.vector, b.vector, α, β)
end

# assign! updates the values in one ErbarVector to the match the values of another
# think of this like c = a, except instead of pointing to the same array, this is more like a
# deep copy
function assign!(c::ErbarVector, a::ErbarVector)
    c.ρ .= a.ρ
    c.m .= a.m
    c.θ .= a.θ
    c.ρ_minus .= a.ρ_minus
    c.ρ_plus .= a.ρ_plus
    c.ρ_avg .= a.ρ_avg
    c.q .= a.q
end


# assign! assign!s the vectors of two bundles
function assign!(c::ErbarBundle, a::ErbarBundle)
    assign!(c.vector, a.vector)
end

# compute the inner product of two vectors in H
function hdot(u::ErbarBundle, v::ErbarBundle)
    N = u.cache.N
    Q = u.cache.Q
    π = u.cache.π
    h = 1 / N
    Qprime = reshape(Q, 1, size(Q) ...)
    πprime = reshape(π, 1, :, 1)
    ρ_sum = sum(u.vector.ρ .* v.vector.ρ * π)
    ρ_avg_sum = sum(u.vector.ρ .* v.vector.ρ * π)
    q_sum = sum(u.vector.q .* v.vector.q * π)
    m_sum = 0.5 * sum(u.vector.m .* v.vector.m .* Qprime .* πprime)
    θ_sum = 0.5 * sum(u.vector.θ .* v.vector.θ .* Qprime .* πprime)
    ρ_min_sum = 0.5 * sum(u.vector.ρ_minus .* v.vector.ρ_minus .* Qprime .* πprime)
    ρ_plus_sum = 0.5 * sum(u.vector.ρ_plus .* v.vector.ρ_plus .* Qprime .* πprime)
    return h * (ρ_sum + ρ_avg_sum + q_sum + m_sum + θ_sum + ρ_min_sum + ρ_plus_sum)
end

# compute the norm of two vectors in H
function norm(u::ErbarBundle)
    return sqrt(hdot(u,u))
end

# compute the action of the discrete curve encoded in a EB u
# cf. eqn (18) in Erbar et al 2020
# TODO: clean this up so that it doesn't work via scalar indexing
function action(u::ErbarBundle)
    m = u.vector.m
    θ = u.vector.θ
    Q = convert(Array,u.cache.Q)
    π = u.cache.π
    N, _, _ = size(m)
    h = 1 / N

    integrand = @. ifelse(m == 0, ifelse(θ == 0, 0, Inf), m^2 / θ)
    action = (h / 2) * sum([sum(integrand[i,:,:] .* Q .* π) for i=1:N])

    return action
end

## you can add vectors together with the expression
## c = a + b
## this produces an entirely new vector
#import Base: +
#function +(u::ErbarVector, v::ErbarVector)
    ## if the defining features of the cache differ, it's not legitimate to add u and v
    #ErbarVector(
        #u.ρ + v.ρ,
        #u.m + v.m,
        #u.θ + v.θ,
        #u.ρ_minus + v.ρ_minus,
        #u.ρ_plus + v.ρ_plus,
        #u.ρ_avg + v.ρ_avg,
        #u.q + v.q,
    #)
#end
#
## you can add bundle together with the expression
## c = a + b
## this produces an entirely new bundle
#function +(u::ErbarBundle, v::ErbarBundle)
	#ErbarBundle(u.cache, u.vector + v.vector)
#end
#
## add!(c, a, b)
#function add!(c::ErbarVector, a::ErbarVector, b::ErbarVector)
    #c.ρ = a.ρ .+ b.ρ
    #c.m = a.m .+ b.m
    #c.θ = a.θ .+ b.θ
    #c.ρ_minus = a.ρ_minus .+ b.ρ_minus
    #c.ρ_plus = a.ρ_plus .+ b.ρ_plus
    #c.ρ_avg = a.ρ_avg .+ b.ρ_avg
    #c.q = a.q .+ b.q
#end
#
#function add!(c::ErbarBundle, a::ErbarBundle, b::ErbarBundle)
    #add!(c.vector, a.vector, b.vector)
#end
#
#
#import Base: *
#function *(α::Number, v::ErbarVector)
    #ErbarVector(
        #α * v.ρ,
        #α * v.m,
        #α * v.θ,
        #α * v.ρ_minus,
        #α * v.ρ_plus,
        #α * v.ρ_avg,
        #α * v.q,
    #)
#end
#
#function scale!(α::Number, v::ErbarVector)
    #v.ρ = α .* v.ρ
    #v.m = α .* v.m
    #v.θ=  α .* v.θ
    #v.ρ_minus = α .* v.ρ_minus
    #v.ρ_plus = α .* v.ρ_plus
    #v.ρ_avg = α .* v.ρ_avg
    #v.q = α .* v.q
#end
#
#function scale!(α::Number, B::ErbarBundle)
    #scale!(α, B.vector)
#end
#
#function *(α::Number, B::ErbarBundle)
	#ErbarBundle(B.cache, α * B.vector)
#end
#
#import Base: -
#function -(u::ErbarVector, v::ErbarVector)
    #return u + (-1) * v
#end
#
#function -(u::ErbarBundle, v::ErbarBundle)
    #ErbarBundle(u.cache, u.vector - v.vector)
#end
#
