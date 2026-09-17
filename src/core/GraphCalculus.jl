# various utility functions that don't belong to any particular subroutine

"""
    is_distribution(μ, π) -> Bool

Return `true` if `μ` is a probability density with respect to the steady state `π`,
i.e. `μ ⋅ π == 1`. Asserts that `μ` and `π` have the same length.
"""
function is_distribution(μ, π)
    @assert size(μ, 1) == size(π, 1)
    return μ ⋅ π == 1
end

## ADMISSIBLE MEANS

"""
    geomean(x, y) -> Real

Geometric mean of real numbers `x` and `y`.  Returns `-Inf` if either argument
is negative (convention used by the graph Wasserstein metric tensor).
"""
function geomean(x,y)
	if x < 0 || y < 0
        return -Inf
    else
        return sqrt(x * y)
    end
end

"""
    logmean(s, t; tol=1e-5) -> Real

Logarithmic mean of real numbers `s` and `t`:

    L(s, t) = (s - t) / (log s - log t)

Returns `-Inf` if either argument is negative, `0` if either is zero and the
other non-negative, and `(s + t) / 2` (first-order Taylor approximation) when
`|s - t| ≤ tol`.
"""
function logmean(s, t; tol=1e-5)
    if minimum([s t]) < 0
        return -Inf
	elseif abs(s-t) ≤ tol
        return (s + t) / 2 # believe it or not, that's the first order taylor expansion near the diagonal...
    else
        return (s - t) / (log(s) - log(t))
    end
end

# partial derivatives of logmean
"""
    logmean_partial_s(s, t; tol=1e-5)

Partial derivative of the logarithmic mean with respect to `s`.

For `|s - t| < tol` (near the diagonal), returns the limiting value 0.5.
Otherwise returns the exact derivative:

    ∂/∂s[(s - t)/(log s - log t)] = (-s + t + s·log(s/t)) / (s·(log s - log t)²)
"""
logmean_partial_s(s, t; tol=1e-5) = abs(s - t) < tol ? 0.5 : (-s + t + s*log(s) - s*log(t))/(s*(log(s) - log(t))^2)

"""
    logmean_partial_t(s, t; tol=1e-5)

Partial derivative of the logarithmic mean with respect to `t`.

For `|s - t| < tol` (near the diagonal), returns the limiting value 0.5.
Otherwise returns the exact derivative:

    ∂/∂t[(s - t)/(log s - log t)] = (s - t - t·log(s/t)) / (t·(log s - log t)²)
"""
logmean_partial_t(s, t; tol=1e-5) = abs(s - t) < tol ? 0.5 : (s - t - t*log(s) + t*log(t))/(t*(log(s) - log(t))^2)


## GRAPH CALCULUS

"""
    graph_gradient(Q, f) -> Matrix

Compute the graph gradient ∇_G(f), where G is the graph induced by the
transition matrix `Q` and `f` is a function defined on the nodes.

Entry `(i, j)` of the result is `f[i] - f[j]` when the edge `(i,j)` exists
(i.e. `Q[i,j] ≠ 0`) and zero otherwise.  The matrix is antisymmetric.
Asserts that `Q` and `f` have compatible dimensions.
"""
function graph_gradient(Q, f)
    @assert size(Q,1) == size(f,1)
    ∇f = zeros(size(Q))
    V = size(Q,1)
    @inbounds for i in 1:V, j in i+1:V
        if Q[i,j] != 0
            d = f[i] - f[j]
            ∇f[i,j] = d
            ∇f[j,i] = -d
        end
    end
    return ∇f
end

"""
    add_graph_gradient!(dest, Q, f)

Add the graph gradient ∇_G(f) directly into `dest` in-place.
Equivalent to `dest .+= graph_gradient(Q, f)` but with no intermediate
array allocation: the loop body writes only to edge positions and skips
non-edges via the sparsity check on Q.
"""
function add_graph_gradient!(dest, Q, f)
    V = size(Q, 1)
    @inbounds for i in 1:V, j in i+1:V
        if Q[i,j] != 0
            d = f[i] - f[j]
            dest[i,j] += d
            dest[j,i] -= d
        end
    end
end

"""
    graph_divergence(Q, m) -> Vector

Compute the graph divergence of the V×V edge field `m` with respect to the
graph induced by `Q`.  Entry `i` of the result is

    (div m)[i] = (1/2) · Σ_j Q[i,j] · (m[j,i] - m[i,j])
"""
function graph_divergence(Q, m)
    V = size(Q, 1)
    div = zeros(V)
    @inbounds for i in 1:V
        s = 0.0
        for j in 1:V
            s += Q[i,j] * (m[j,i] - m[i,j])
        end
        div[i] = 0.5 * s
    end
    return div
end

"""
    graph_divergence!(out, Q, m_t)

Compute the graph divergence of the V×V edge field `m_t` into the
pre-allocated length-V vector `out`, with no allocations.
"""
function graph_divergence!(out, Q, m_t)
    V = size(Q, 1)
    @inbounds for i in 1:V
        s = 0.0
        for j in 1:V
            s += Q[i,j] * (m_t[j,i] - m_t[i,j])
        end
        out[i] = 0.5 * s
    end
end


"""
    laplacian_from_transition(Q) -> Matrix

Return the graph Laplacian `L = Q - D` where `D` is the diagonal degree matrix
`D[i,i] = Σ_j Q[i,j]`.
"""
function laplacian_from_transition(Q)
    return Q - Diagonal(reshape(sum(Q, dims=2), (size(Q, 1),)))
end


"""
    avg_operator(N)

Return the sparse time-averaging operator of size `(N-1) × N`.

Row `i` contains 0.5 at columns `i` and `i+1`, so multiplying by a length-`N`
vector computes the average of each consecutive pair.  In the Erbar et al. 2020
discretisation, this maps the node density curve ρ ∈ V_{n,h}^1 (N+1 time
points) to its time-averaged version ρ_avg ∈ V_{n,h}^0 (N time points) via
`avg_operator(N+1) * ρ`.
"""
function avg_operator(N)
    return sparse(0.5 * Tridiagonal(zeros(N-1), ones(N), ones(N-1))[1:N-1,:])

end

"""
    finite_difference_operator(N)

Return the sparse forward finite-difference operator of size `(N-1) × N`.

Multiplying by a length-`N` vector f returns the vector of forward differences
`[f[2]-f[1], f[3]-f[2], …, f[N]-f[N-1]]`.
"""
function finite_difference_operator(N)
    d = -1. * ones(N)
    u = ones(N-1)
    l = zeros(N-1)
    A = Tridiagonal(l, d ,u)
    return sparse(A[1:N-1,:])
end

"""
    metric_tensor(ρ, mean=geomean)

Return the `V × V` metric tensor matrix for the probability density `ρ`.

Entry `(i, j)` is `mean(ρ[i], ρ[j])` for `i ≠ j` and zero on the diagonal.
The default mean is the geometric mean, giving the Riemannian metric associated
to the graph Wasserstein geometry of Erbar et al. 2020.  Passing `mean=logmean`
gives the metric for the logarithmic-mean variant.
"""
function metric_tensor(ρ, mean=geomean)
    N = size(ρ, 1)
    g = zeros(N,N)
    for i in 1:N
        for j in i+1:N
            g[i, j] = mean(ρ[i], ρ[j])
            g[j, i] = mean(ρ[i], ρ[j])
        end
    end
    return g
end
