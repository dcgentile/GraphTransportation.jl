"""
    markov_chain_from_edge_list(E) -> (Q, π)

Construct the uniform random walk Markov chain on the graph defined by the
edge list `E`.  Each element of `E` is a tuple `(i, j)` representing an
undirected edge.  All edges are given equal weight (unweighted graph), so the
transition probability from node `x` to each neighbor is `1 / degree(x)`.
The stationary distribution satisfies `π[x] ∝ degree(x)`.

Returns `(Q, π)` where `Q` is the row-stochastic transition matrix and `π` is
the stationary distribution vector.
"""
function markov_chain_from_edge_list(E)
    V = maximum(maximum(E))
    A = zeros(V,V)

    for e in E
        i, j = e
        A[i,j] = 1
        A[j,i] = 1
    end

    Q, π = markov_chain_from_weight_matrix(A)

    @assert Q' * π ≈ π
    return (Q, π)
end

"""
    markov_chain_from_adjacency_matrix(A) -> (Q, π)

Construct the uniform random walk Markov chain from a binary adjacency matrix
`A`.  Equivalent to `markov_chain_from_weight_matrix(A)`.  The transition
matrix is `Q[x,y] = A[x,y] / degree(x)` and the stationary distribution is
`π[x] ∝ degree(x)`.

Returns `(Q, π)`.
"""
function markov_chain_from_adjacency_matrix(A::Matrix)
    V, _ = size(A)
    S = sparse(A)
    E = nnz(S)
    d = A * ones(V)
    π = d / E
    Q = A ./ d

    @assert Q' * π ≈ π

    return (Q, π)
end

"""
    markov_chain_from_weight_matrix(W) -> (Q, π)

Construct a weighted random walk Markov chain from a non-negative symmetric
weight matrix `W`.  The transition probability from node `x` to node `y` is
`Q[x,y] = W[x,y] / sum(W[x,:])`, and the stationary distribution is
`π[x] = sum(W[x,:]) / sum(W)`.

Returns `(Q, π)`.
"""
function markov_chain_from_weight_matrix(W::Matrix)
    d = vec(sum(W, dims=2))
    Q = W ./ reshape(d, : ,1)
    π = d / sum(d)

    @assert Q' * π ≈ π

    return (Q, π)
end

"""
    stationary_from_transition(Q) -> π

Compute the stationary distribution of a row-stochastic transition matrix `Q`
by solving the linear system `(Q' - I)π = 0` with the normalisation constraint
`sum(π) = 1`.  The overdetermined system is assembled and solved via least
squares (`\\`).
"""
function stationary_from_transition(Q)
    n = size(Q, 1)
    A = [Q' - I; ones(1, n)]
    b = [zeros(n); 1.0]
    v = A \ b
    return v / sum(v)
end
