using LinearAlgebra

# various utility functions that don't belong to any particular subroutine

# check that a vector μ is a probability density w.r.t the steady state π
function is_distribution(μ, π)
    """
    check to see if a given vector μ is a probability distribution
    w.r.t the steady state π
    raises an error if μ and π are not defined for the same number of nodes
    """
    @assert size(μ, 1) == size(π, 1)
    return μ ⋅ π == 1
end

## ADMISSIBLE MEANS

function geomean(x,y)
    """
    compute the geometric mean of real numbers x, y
    if either is negative, their geometric mean is taken to be -∞
    """
	if x < 0 || y < 0
        return -Inf
    else
        return sqrt(x * y)
    end
end

function logmean(s, t; tol=1e-10)
    """
    compute the logarithmic mean of real numbers s, t
    if either is negative, their logmean is -∞
    if at least one is 0 and the other non-negative, their logmean is 0
    """
    if minimum([s t]) < 0
        return -Inf
	elseif abs(s-t) ≤ tol
        return (s + t) / 2 # believe it or not, that's the first order taylor expansion near the diagonal...
    else
        return (s - t) / (log(s) - log(t))
    end
end

# partial derivatives of logmean
logmean_partial_s(s, t; tol=1e-10) = abs(s - t) < tol ? 0.5 : (-s + t + s*log(s) - s*log(t))/(s*(log(s) - log(t))^2)
logmean_partial_t(s, t; tol=1e-10) = abs(s - t) < tol ? 0.5 : (s - t - t*log(s) + t*log(t))/(t*(log(s) - log(t))^2)


## GRAPH CALCULUS

function graph_gradient(Q, f)
    """
    given a transition rate matrix Q and a function defined on the nodes f
    compute the ∇_G(f), where G is the graph induced by Q
    raises an error if dimensions of Q and f are incompatible
    """
    @assert size(Q,1) == size(f,1)
    ∇f = zeros(size(Q))
    N = size(Q,1)
    for i = 1:N
        for j = i:N
            ∇f[i,j] = Q[i,j] != 0 ? f[i] - f[j] : 0
            ∇f[j,i] = -∇f[i,j]
        end
    end
    return ∇f
end

function graph_divergence(Q, m)
    """
    given a transition rate matrix Q and a vector field m, compute the
    graph divergence of m w.r.t to the graph induced by Q
    TODO: clean this up so that it doesn't work via indexing
    """
    V = size(Q, 1)
    div = zeros(1,V)
    for i in 1:V
        div[1,i] = 0.5 * sum(Q[i,:] .* (m[:, i] .- m[i,:]))
    end
    return div
end


function laplacian_from_transition(Q)
    """
    form the Laplacian matrix of Q
    """
    return Q - Diagonal(reshape(sum(Q, dims=2), (size(Q, 1),)))
end


function avg_operator(N)
    return sparse(0.5 * Tridiagonal(zeros(N-1), ones(N), ones(N-1))[1:N-1,:])

end

function finite_difference_operator(N)
    d = -1. * ones(N)
    u = ones(N-1)
    l = zeros(N-1)
    A = Tridiagonal(l, d ,u)
    return sparse(A[1:N-1,:])
end
