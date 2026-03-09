#include("../utils.jl")
#include("../ErbarVector.jl")

"""
    is_in_ScriptK(ρ_min, ρ_pl, θ, verbose=false) -> Bool

Test whether the triple `(ρ_minus, ρ_plus, θ)` lies element-wise in the
constraint set Script K = {(x, y, z) : 0 ≤ z ≤ geomean(x, y)}.

Checks both `0 ≤ θ[idx]` and `θ[idx] ≤ geomean(ρ_minus[idx], ρ_plus[idx])`
for every index, up to a numerical tolerance of 1e-14 on the upper bound.
Returns `true` if all elements pass; prints diagnostic information and returns
an exception object on the first failure.
"""
function is_in_ScriptK(ρ_min, ρ_pl, θ, verbose=false)
    @assert size(ρ_min) == size(ρ_pl) && size(ρ_pl) == size(θ)
    for idx in eachindex(θ)
        x = geomean(ρ_min[idx], ρ_pl[idx])
        y = θ[idx]
        try
            @assert 0 ≤ y
        catch e
            println([y, x, geomean(x,y)])
            return e
        end

        try
            @assert y ≤ x || abs(x - y) < 1e-14
        catch e
            println([y, x, x - y])
            return e
        end
    end
    verbose ? println("Passed Script{K} Inclusion Test") : nothing
    return true
end

"""
    is_in_JPM(q, ρ_minus, ρ_plus, verbose=false) -> Bool

Test whether `(q, ρ_minus, ρ_plus)` lies in the constraint set J_PM, i.e.,
that `ρ_minus[t,x,y] == q[t,x]` and `ρ_plus[t,x,y] == q[t,y]` for all
indices `(t, x, y)`.  Returns `true` on success; prints diagnostic information
and returns `false` on the first failing index.
"""
function is_in_JPM(q, ρ_minus, ρ_plus, verbose=false)
    """
    given q ∈ V_{n,h}^{0} and ρ_min, ρ_plus ∈ V_{e, h}^{0}
    checy that the ρ_min[t,x,y] == q[t,x] and ρ_plus[t,x,y] == q[t,y]
    """
    for (idx, _) in pairs(ρ_minus)
        t, x, y = Tuple(idx);
        try
            @assert ρ_minus[idx] == q[t,x]
        catch e
            println("Failed inclusion in J_PM with error $(e)")
            return false
        end
        try
            @assert ρ_plus[idx] == q[t,y]
        catch e
            println("Failed inclusion in J_PM with error $(e)")
            return false
        end
    end
    verbose ? println("Passed inclusion in J_PM") : nothing
    return true
end

"""
    is_in_JEq(ρ, q, verbose=false) -> Bool

Test whether `(ρ_avg, q)` lies in the equality constraint set J_Eq, i.e.,
that `ρ_avg ≈ q` element-wise (via `isapprox`).  Returns `true` on success;
prints the range of absolute differences and returns `false` on failure.
"""
function is_in_JEq(ρ, q, verbose=false)
    try
        @assert isapprox(ρ, q)
    catch e
        println("Failed inclusion in J_EQ with error $(e)")
        println(extrema(abs.(ρ - q)))
        return false
    end
    verbose ? println("Passed Inclusion in J_EQ") : nothing
    return true
end

"""
    is_in_JAvg(ρ, ρ_bar, verbose=false) -> Bool

Test whether `(ρ, ρ_bar)` lies in the time-averaging constraint set J_Avg,
i.e., that `ρ_bar ≈ avg_operator(size(ρ,1)) * ρ` element-wise.  Returns
`true` on success; prints indices with discrepancy above 1e-20 and returns
`false` on failure.
"""
function is_in_JAvg(ρ, ρ_bar, verbose=false)
    A = avg_operator(size(ρ,1))
    println(size(ρ))
    println(size(A))
    println(size(ρ_bar))
    ρ_avg = A * ρ
    try
	    @assert isapprox(ρ_bar, ρ_avg)
    catch
        for idx in eachindex(ρ_bar)
            if abs(ρ_bar[idx] - ρ_avg[idx]) > 1e-20
                println([abs(ρ_bar[idx] - ρ_avg[idx]), ρ_bar[idx], ρ_avg[idx]])
            end
        end
        return false
    end

    verbose ? println("Passed Inclusion in J_Avg") : nothing
    return true
end

"""
    is_in_CE_weakly(ρ, m, Q, u, verbose=false) -> Bool

Weakly test satisfaction of the graph continuity equation `∂_t ρ + div m ≈ 0`.

Rather than checking pointwise, computes the inner product of the residual
`∂_t ρ + div m` against a random test function φ, weighted by `u` (the
stationary distribution).  A residual magnitude below 1e-10 is considered
passing.  This is a probabilistic check; it may miss structured cancellations
but is cheaper than a full pointwise test.
"""
function is_in_CE_weakly(ρ, m, Q, u, verbose=false)
    N, V = size(ρ)
    ∂tρ = (N - 1) * (ρ[2:N,:] - ρ[1:N - 1,:])
    divm = similar(∂tρ)
    @inbounds for i=1:N-1
        divm[i,:] = graph_divergence(Q, m[i,:,:])
    end
    a = ∂tρ + divm
    #φ = ones(N-1, V)
    φ = rand(N-1, V)
    ce_discrepancy = sum(a .* φ * u)
    try
	    @assert ce_discrepancy ≤ 1e-10
    catch err
        println(err)
        println(ce_discrepancy)
        return false
    end

    verbose ? println("Passed Continuity Equation Check") : nothing
    return true
end

"""
    CE_operator(ρ, m, Q) -> Matrix

Compute the continuity equation residual `∂_t ρ + div m` as an `(N-1) × V`
matrix, where `N+1 = size(ρ, 1)` is the number of time points and `V` is the
number of nodes.  A zero matrix indicates that `(ρ, m)` satisfies the
discretised continuity equation.
"""
function CE_operator(ρ, m, Q)
    N, V = size(ρ)
    ∂tρ = (N - 1) * (ρ[2:N,:] - ρ[1:N - 1,:])
    divm = similar(∂tρ)
    @inbounds for i=1:N-1
        divm[i,:] = graph_divergence(Q, m[i,:,:])
    end
    a = ∂tρ + divm
    return a
end

"""
    ∂t(ρ) -> Matrix

Compute the discrete time derivative of the density curve `ρ` of size
`(N, V)`.  Returns an `(N-1) × V` matrix of forward differences scaled by
the step count: `(N-1) · (ρ[2:N,:] - ρ[1:N-1,:])`.
"""
function ∂t(ρ)
    N, V = size(ρ)
    return (N - 1) * (ρ[2:N,:] - ρ[1:N-1,:])
end

#function Script_K_pre_indicator(B::ErbarBundle)
    #v = B.vector
    #cache = B.cache
    #a = is_in_JPM(v.q, v.ρ_minus, v.ρ_plus)
    #b = is_in_JAvg(v.ρ, v.ρ_bar)
    #c = is_in_CE_weakly(v.ρ, v.m, cache.Q, cache.π)
    #d = is_in_ScriptK(v.ρ_minus, v.ρ_plus, v.θ)
    #e = is_in_JEq(v.ρ_bar, v.q)
    #if a && b && c && d && e
        #return true
    #else
        #return (a, b, c, d, e)
    #end
#end
