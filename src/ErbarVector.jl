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

    function ErbarCache(
        Q::AbstractMatrix,
        π::AbstractVector,
        μ::AbstractVector,
        ν::AbstractVector,
        N::Int
    )
        try
            @assert μ' * π ≈ 1
        catch error
		    @warn "μ is not a density wrt to π, μ ⋅ π = $(μ ⋅ π)"
        end
        try
            @assert ν' * π ≈ 1
        catch error
		    @warn "ν is not a density wrt to π, ν ⋅ π = $(ν ⋅ π)"
        end

        # form the linear systems we'll be solving in each step
        ceh_sys = form_ceh_system(Q, N)
        #avg_sys = factorize(form_avg_system(N))
        avg_sys = form_avg_system(N)

        new(Q, μ, ν, π, N, ceh_sys, avg_sys)

    end

    function ErbarCache(
        Q, μ, ν, N;
    )
        #V, _ = size(Q)
        #S = sparse(Q)
        #E = nnz(S)
        #π = zeros(V)
        #for i in 1:V
            #π[i] = nnz(S[i,:]) / E
        #end

        function stationary(Q)
            n = size(Q, 1)
            A = [Q' - I; ones(1, n)]
            b = [zeros(n); 1.0]
            v = A \ b
            return v / sum(v)
        end

        π = stationary(Q)
        
        try
            @assert μ' * π ≈ 1
        catch error
		    @warn "μ is not a density wrt to π, we have μ ⋅ π = $(μ ⋅ π)"
        end
        try
            @assert ν' * π ≈ 1
        catch error
		    @warn "ν is not a density wrt to π, we have ν ⋅ π = $(ν ⋅ π)"
        end

        # form the linear systems we'll be solving in each step
        ceh_sys = form_ceh_system(Q, N)
        avg_sys = form_avg_system(N)

        new(sparse(Q), μ, ν, π, N, ceh_sys, avg_sys)
    end

    # Cheap constructor for warm-starting: reuse ceh_sys and avg_sys from an
    # existing cache (they depend only on Q and N, not on μ or ν) and only
    # update the boundary conditions.
    function ErbarCache(existing::ErbarCache, μ_new::AbstractVector, ν_new::AbstractVector)
        new(existing.Q, μ_new, ν_new, existing.π, existing.N, existing.ceh_sys, existing.avg_sys)
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

    function ErbarBundle(
        Q::AbstractMatrix, μ::AbstractVector, ν::AbstractVector, N::Int;
    )
        cache = ErbarCache(Q, μ, ν, N)

        #### INITIALIZE VECTOR COMPONENTS
        V, _ = size(Q)
        ρ = zeros(N + 1, V)
        for i=1:N+1
            t = (i - 1) / N
            ρ[i,:] = (1 - t) * μ + t * ν
        end
        m = zeros(N, V, V)
        θ = zeros(N, V, V)
        ρ_minus = zeros(N, V, V)
        ρ_plus = zeros(N, V, V)
        ρ_avg = zeros(N,V)
        q = zeros(N,V)

        vector = ErbarVector(ρ, m ,θ, ρ_minus, ρ_plus, ρ_avg, q)
        new(cache, vector)

    end

    function ErbarBundle(
        Q::AbstractMatrix,
        steady_state::AbstractVector,
        μ::AbstractVector,
        ν::AbstractVector,
        N::Int;
    )
        cache = ErbarCache(Q, steady_state, μ, ν, N)

        #### INITIALIZE VECTOR COMPONENTS
        V, _ = size(Q)
        ρ = zeros(N + 1, V)
        for i=1:N+1
            t = (i - 1) / N
            ρ[i,:] = (1 - t) * μ + t * ν
        end
        m = zeros(N, V, V)
        θ = zeros(N, V, V)
        ρ_minus = zeros(N, V, V)
        ρ_plus = zeros(N, V, V)
        ρ_avg = zeros(N,V)
        q = zeros(N,V)
        vector = ErbarVector(ρ, m ,θ, ρ_minus, ρ_plus, ρ_avg, q)
        new(cache, vector)

    end
end


Base.copy(a::ErbarCache) = ErbarCache(a, a.μ, a.ν)
Base.copy(a::ErbarVector) = ErbarVector(copy(a.ρ), copy(a.m), copy(a.θ), copy(a.ρ_minus), copy(a.ρ_plus), copy(a.ρ_avg), copy(a.q))
Base.copy(a::ErbarBundle) = ErbarBundle(copy(a.cache), copy(a.vector))


# combine! computes a linear combination of ErbarVectors and stores the result in a
# pre-allocated vector c
"""
    combine!(c::ErbarVector, a::ErbarVector, b::ErbarVector, α::Number, β::Number)

Description of the function.

#TODO
"""
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
"""
    combine!(c::ErbarBundle, a::ErbarBundle, b::ErbarBundle, α::Number, β::Number)

Description of the function.

#TODO
"""
function combine!(c::ErbarBundle, a::ErbarBundle, b::ErbarBundle, α::Number, β::Number)
    combine!(c.vector, a.vector, b.vector, α, β)
end

# assign! updates the values in one ErbarVector to the match the values of another
# think of this like c = a, except instead of pointing to the same array, this is more like a
# deep copy
"""
    assign!(c::ErbarVector, a::ErbarVector)

Description of the function.

#TODO
"""
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
"""
    assign!(c::ErbarBundle, a::ErbarBundle)

Description of the function.

#TODO
"""
function assign!(c::ErbarBundle, a::ErbarBundle)
    assign!(c.vector, a.vector)
end

# compute the action of the discrete curve encoded in a EB u
# cf. eqn (18) in Erbar et al 2020
"""
    action(u::ErbarBundle)

Description of the function.

#TODO
"""
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

