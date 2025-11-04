"""
    step_direction(ν, M, weights, Q)

REQUIRED ARGS
Q: Markov transition rate matrix representing the underlying graph
ν: probability measure w.r.t. the steady state of Q
M: array of size (num_nodes, num_measures). Each column should be a probability measure w.r.t to the steady state of Q
weights: a non-negative vector of size num_measures with sum(weights) == 1

OPTIONAL ARGS:
tol: positive float, convergence threshold
n_steps: integer, determines how many steps are used for computing the geodesics which yield the tangent vector

"""
function step_direction(ν, M, weights, Q; sstate=nothing, tol=1e-10, n_steps=100)
    tangent_vector = zeros(size(Q))
    p = size(M, 2)
    for i=1:p
        gamma, _ = isnothing(sstate) ? BBD(Q, ν, M[:, i], N = n_steps, tol=tol) : BBD(Q, sstate, ν, M[:, i], N = n_steps, tol=tol)
        tangent_vector = tangent_vector + weights[i] * (gamma.vector.m[0,:,:])
    end
    return tangent_vector
end


"""
    barycenter(M, weights, Q; h=0.1, maxiters=100, tol=1e-8, geodesic_tol=1e-10, geodesic_steps=100)

REQUIRED ARGS
Q: Markov transition rate matrix representing the underlying graph
M: array of size (num_nodes, num_measures). Each column should be a probability measure w.r.t to the steady state of Q
weights: a non-negative vector of size num_measures with sum(weights) == 1

OPTIONAL ARGS:
h: positive float, step size for Wasserstein gradient descent scheme
maxiters: positive integer, cap on number of iterations for scheme
tol: positive float, convergence threshold
geodesic_tol: positive float, convergence threshold for computing the geodesics which yield the tangent vectors
geodesic_steps: integer, determines how many steps are used for computing the geodesics which yield the tangent vector


#TODO
"""
function barycenter(M, weights, Q;
                    sstate=nothing,
                    h=0.1, maxiters=100, tol=1e-8, geodesic_tol=1e-10, geodesic_steps=100)
    ν_old = ones(size(Q,1)) # inital condition of flow
    ν_new = ones(size(Q,1)) # inital condition of flow
    for _=1:maxiters
        δJ = step_direction(ν_old, M, weights, Q, sstate=sstate, tol=geodesic_tol, n_steps=geodesic_steps)
        ν_new .= ν_old .+ h * graph_divergence(Q, metric_tensor(ν_old) .* δJ)
        if sqrt(sum((ν_new - ν_old).^2)) < tol
            break
        else 
            ν_old = ν_new
        end
    end
    return ν_new
end


"""
    coordinates(ν, M, Q, maxiters=100, tol=1e-8, h=0.01)

Description of the function.

#TODO
"""
function coordinates(ν, M, Q, maxiters=100, tol=1e-8, h=0.01)
    error("Not Implemented")
end
