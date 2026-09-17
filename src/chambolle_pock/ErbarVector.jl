"""
    ErbarVector{M, T}

An element of the Hilbert space H described in equation (27) of Erbar et al.
2020, representing a discretised curve `(ρ, m)` together with the auxiliary
variables `(θ, ρ_minus, ρ_plus, ρ_avg, q)` used by the Chambolle-Pock solver.

Type parameters:
- `M`: concrete 2-D array type for node fields (`ρ`, `ρ_avg`, `q`); typically `Matrix{Float64}`
- `T`: concrete 3-D array type for edge fields (`m`, `θ`, `ρ_minus`, `ρ_plus`); typically `Array{Float64,3}`

Parameterising on the concrete types allows Julia to fully specialise all
operations on `ErbarVector` fields in the Chambolle-Pock inner loop.

# Fields
- `ρ`: density curve, shape `(N+1) × V`
- `m`: edge flux, shape `N × V × V`
- `θ`: dual edge variable, shape `N × V × V`
- `ρ_minus`, `ρ_plus`: edge density copies, shape `N × V × V`
- `ρ_avg`: time-averaged density, shape `N × V`
- `q`: dual node variable, shape `N × V`
"""
mutable struct ErbarVector{M<:AbstractMatrix{Float64}, T<:AbstractArray{Float64,3}}
    # vector components to operate on
    ρ::M
    m::T
    θ::T
    ρ_minus::T
    ρ_plus::T
    ρ_avg::M
    q::M
end

"""
    ErbarCache

Immutable container for the graph OT problem metadata: everything that needs
to be computed only once given `(Q, μ, ν, N)`.

Constructors:
- `ErbarCache(Q, π, μ, ν, N)` — use a pre-computed stationary distribution `π`
- `ErbarCache(Q, μ, ν, N)` — computes `π` internally
- `ErbarCache(existing, μ_new, ν_new)` — cheap warm-start copy reusing the cached
  linear-system factorisations (`ceh_sys`, `avg_sys` depend only on `Q` and `N`)

# Fields
- `Q`: Markov transition matrix
- `μ`, `ν`: boundary probability densities
- `π`: stationary distribution of `Q`
- `N`: number of geodesic time steps
- `ceh_sys`: factorised continuity-equation linear system
- `avg_sys`: factorised time-averaging linear system
"""
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

"""
    ErbarBundle{M, T}

A pair `(cache::ErbarCache, vector::ErbarVector)` representing the complete
state of a graph OT problem: the problem metadata and the current iterate.

Type parameters `{M, T}` mirror those of `ErbarVector` so that accessing
`bundle.vector` in hot-path code returns a concretely-typed value, enabling
full specialisation of the Chambolle-Pock inner loop without dynamic dispatch.

Constructors:
- `ErbarBundle(Q, μ, ν, N)` — compute `π` internally
- `ErbarBundle(Q, π, μ, ν, N)` — use pre-computed stationary distribution
- `ErbarBundle(cache, vector)` — assemble from existing components
"""
mutable struct ErbarBundle{M<:AbstractMatrix{Float64}, T<:AbstractArray{Float64,3}}
    cache::ErbarCache
    vector::ErbarVector{M,T}

    function ErbarBundle(cache::ErbarCache, vector::ErbarVector{M,T}) where {M,T}
        new{M,T}(cache, vector)
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

        vector = ErbarVector(ρ, m, θ, ρ_minus, ρ_plus, ρ_avg, q)
        new{Matrix{Float64}, Array{Float64,3}}(cache, vector)

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

        vector = ErbarVector(ρ, m, θ, ρ_minus, ρ_plus, ρ_avg, q)
        new{Matrix{Float64}, Array{Float64,3}}(cache, vector)

    end
end


Base.copy(a::ErbarCache) = ErbarCache(a, a.μ, a.ν)
Base.copy(a::ErbarVector) = ErbarVector(copy(a.ρ), copy(a.m), copy(a.θ), copy(a.ρ_minus), copy(a.ρ_plus), copy(a.ρ_avg), copy(a.q))
Base.copy(a::ErbarBundle) = ErbarBundle(copy(a.cache), copy(a.vector))


# combine! computes a linear combination of ErbarVectors and stores the result in a
# pre-allocated vector c
"""
    combine!(c::ErbarVector, a::ErbarVector, b::ErbarVector, α::Number, β::Number)

Compute the linear combination `c = α·a + β·b` in-place across all seven
component arrays of the ErbarVector (ρ, m, θ, ρ_minus, ρ_plus, ρ_avg, q).
No intermediate arrays are allocated; all operations use broadcasting with `.=`.
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

Bundle-level wrapper for `combine!` on ErbarVectors.  Computes `c.vector =
α·a.vector + β·b.vector` in-place; the cache fields of the bundles are not
modified.
"""
function combine!(c::ErbarBundle, a::ErbarBundle, b::ErbarBundle, α::Number, β::Number)
    combine!(c.vector, a.vector, b.vector, α, β)
end

# assign! updates the values in one ErbarVector to the match the values of another
# think of this like c = a, except instead of pointing to the same array, this is more like a
# deep copy
"""
    assign!(c::ErbarVector, a::ErbarVector)

Copy all component arrays of `a` into `c` in-place (deep value copy, not
pointer aliasing).  Equivalent to `c = copy(a)` but reuses `c`'s existing
allocations.  Covers all seven fields: ρ, m, θ, ρ_minus, ρ_plus, ρ_avg, q.
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

Bundle-level wrapper for `assign!` on ErbarVectors.  Copies `a.vector` into
`c.vector` in-place; the cache fields are not modified.
"""
function assign!(c::ErbarBundle, a::ErbarBundle)
    assign!(c.vector, a.vector)
end

# compute the action of the discrete curve encoded in a EB u
# cf. eqn (18) in Erbar et al 2020
"""
    action(u::ErbarBundle) -> Float64

Compute the discrete action functional of the curve encoded in `u`, as defined
in equation (18) of Erbar et al. 2020:

    A(ρ, m) = (h/2) · Σ_t Σ_{x,y} (m[t,x,y]² / θ[t,x,y]) · Q[x,y] · π[x]

where `h = 1/N` is the time step.  The integrand uses the convention
`m²/θ = 0` when `m = θ = 0`, and `+Inf` when `m ≠ 0` and `θ = 0`.

The square root of the action is the discrete Wasserstein distance between the
boundary measures `u.cache.μ` and `u.cache.ν`.
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

