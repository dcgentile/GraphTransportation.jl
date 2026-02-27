function markov_chain_from_edge_list(E)
    V = maximum(maximum(E)) 
    A = zeros(V,V)
    
    for e in E
        i, j = e
        A[i,j] = 1
        A[j,i] = 1
    end

    Q, π = markov_chain_from_weight_matrix(A)

    @assert Q' * π == π
    return (Q, π)
end

function markov_chain_from_adjacency_matrix(A::Matrix)
    V, _ = size(A)
    S = sparse(A)
    E = nnz(S)
    d = A * ones(V)
    π = d / E
    Q = A ./ d

    @assert Q' * π == π
    return (Q, π)
end

function markov_chain_from_weight_matrix(W::Matrix)
    Q = similar(W)
    V, = size(W)

    d = W * ones(V)
    Q = W ./ d
    π = d / sum(d)

    @assert Q' * π == π

    return (Q, π)
end


function stationary_from_transition(Q)
    n = size(Q, 1)
    A = [Q' - I; ones(1, n)]
    b = [zeros(n); 1.0]
    v = A \ b
    return v / sum(v)
end
