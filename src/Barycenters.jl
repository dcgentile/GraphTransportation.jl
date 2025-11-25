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
    variance = 0
    for i=1:p
        gamma, dist = isnothing(sstate) ? BBD(Q, ν, M[:, i], N = n_steps, tol=tol) : BBD(Q, sstate, ν, M[:, i], N = n_steps, tol=tol)
        tangent_vector = tangent_vector + weights[i] * (gamma.vector.m[1,:,:])
        variance = variance + 0.5 * weights[i] * dist^2
    end
    println(variance)
    return tangent_vector, variance
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
                    initialization=nothing,
                    h=0.1, maxiters=100, tol=1e-8, geodesic_tol=1e-10, geodesic_steps=100)

    ν = isnothing(initialization) ? ones(size(Q,1)) : initialization # inital condition of flow
    ν_next = copy(ν)

    norm_diffs = []
    variances = []

    for k =1:maxiters

        δJ, variance = step_direction(ν, M, weights, Q, sstate=sstate, tol=geodesic_tol, n_steps=geodesic_steps)
        append!(variances, variance)
        ν_next = ν .- h * graph_divergence(Q, metric_tensor(ν) .* δJ)

        norm_diff = norm(ν_next - ν)
        append!(norm_diffs, norm_diff)

        if norm_diff < tol
            println("finished in $(k) iterations")
            break
        else 
            println("iteration $(k): normdiff=$(norm_diff)")
            println("measure: $(ν_next)")
            ν = ν_next
        end

    end
    return (ν_next, norm_diffs, variances)
end


"""
    coordinates(ν, M, Q)

Description of the function.

#TODO
"""
function analysis(ν, M, Q; N=32, tol=1e-10)
    p = size(M, 2)

    # compute the tangent vectors of the geodesics from the target measure to each reference
    tangent_vectors = [BBD(Q, ν, M[:,i], N=N, tol=tol)[1].vector.m[1,:,:] for i=1:p]
    g = metric_tensor(ν)

    # form the matrix A for the QP
    A = zeros(p,p)
    for i=1:p, j=i:p
        A[i,j] = A[j,i] = sum(tangent_vectors[i] .* tangent_vectors[j] .* g)
    end

    println(A)

    # solve the QP
    n = size(A, 1)
    x = Variable(n)
    problem = minimize(quadform(x, A))
    # Simplex constraints
    problem.constraints = vcat(problem.constraints, [x >= 0])
    problem.constraints = vcat(problem.constraints, [sum(x) == 1])
    
    solve!(problem, SCS.Optimizer)
    x.value  # optimal solution
end
