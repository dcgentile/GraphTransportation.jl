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
function step_direction(ν, M, weights, Q; tol=1e-10, n_steps=100, prev_geodesics=nothing)
    tangent_vector = zeros(size(Q))
    p = size(M, 2)
    variance = 0
    geodesics = Vector{Any}(undef, p)
    for i = 1:p
        init  = isnothing(prev_geodesics) ? nothing : prev_geodesics[i]
        gamma = discrete_transport(Q, ν, M[:, i], N=n_steps, tol=tol, initialization=init)
        tangent_vector += weights[i] * gamma.vector.m[1,:,:]
        variance       += 0.5 * weights[i] * action(gamma)^2
        geodesics[i]    = gamma
    end
    return tangent_vector, variance, geodesics
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
                    h=1., maxiters=100, tol=1e-8,
                    geodesic_tol=1e-10, geodesic_steps=100,
                    return_stats=false, initialization_index=1,
                    initialization=nothing, geodesic_warmstart=true, verbose=true)
    ν = isnothing(initialization) ? M[:,initialization_index] : copy(initialization)
    ν_next = copy(ν)

    steady_state = stationary_from_transition(Q)
    root_steady_state = sqrt.(steady_state)

    norm_diffs = zeros(maxiters)
    variances = zeros(maxiters)

    prog = verbose ? Progress(maxiters; desc="WGD ", showspeed=true, color=:cyan) : nothing

    prev_geodesics = nothing

    for k = 1:maxiters

        δJ, variance, geodesics = step_direction(ν, M, weights, Q,
                                      tol=geodesic_tol, n_steps=geodesic_steps,
                                      prev_geodesics=geodesic_warmstart ? prev_geodesics : nothing)
        prev_geodesics = geodesics
        variances[k] = variance

        div_term = graph_divergence(Q, metric_tensor(ν) .* δJ)
        ν_next = ν .- (1 + 0.1 * randn()) * h * div_term

        n_halve = 0
        while abs(dot(ν_next, steady_state) - 1) > 1e-8 || minimum(ν_next) < 0
            n_halve += 1
            n_halve > 3 && error("Step size reduction failed 3 times at iteration $k: " *
                                 "ν_next is not a valid probability measure " *
                                 "(⟨ν_next, u⟩ = $(dot(ν_next, steady_state)))")
            ν_next = ν .- (h / 2^n_halve) * div_term
        end

        norm_diff = norm((ν_next - ν) .* root_steady_state)
        norm_diffs[k] = norm_diff

        if verbose
            ProgressMeter.next!(prog; showvalues=[
                ("‖Δν‖", @sprintf("%.4e", norm_diff)),
                ("var",  @sprintf("%.4f", variance)),
            ])
        end

        if norm_diff < tol
            verbose && ProgressMeter.finish!(prog; desc="WGD (converged) ")
            break
        else
            ν = ν_next
        end

    end
    return return_stats ? (ν_next, norm_diffs, variances) : ν_next
end


"""
    iterated_barycenter(M, weights, Q; steps=[2,4,8,16,32], kwargs...)

Warm-started barycenter computation. Runs `barycenter` repeatedly with the
geodesic step counts in `steps` (doubling schedule by default), using each
stage's output as the initialization for the next. The rationale is that
starting gradient descent near the minimum means the expensive high-accuracy
geodesic stages converge in far fewer iterations.

All additional kwargs are forwarded to `barycenter` at every stage.
"""
function iterated_barycenter(M, weights, Q;
                             steps=[2, 4, 8, 16, 32],
                             kwargs...)
    ν = nothing
    for s in steps
        verbose = get(kwargs, :verbose, true)
        verbose && println("  → geodesic_steps = $s")
        ν = barycenter(M, weights, Q; geodesic_steps=s, initialization=ν, kwargs...)
    end
    return ν
end


"""
    analysis(ν, M, Q; N=100, tol=1e-10, compute_condition=false,
             return_system=false) -> weights

Recover the barycentric coordinates of `ν` with respect to the reference
measures in `M` (a `V × p` matrix whose columns are probability densities
w.r.t. the stationary distribution of `Q`).

The method proceeds in three steps:
  1. For each reference measure `M[:,i]`, compute the geodesic from `ν` to
     `M[:,i]` via `discrete_transport` and extract the initial tangent vector
     `m[1,:,:]` (the logarithmic map at `ν`).
  2. Assemble the `p × p` Gram matrix `A[i,j] = ⟨log_ν(M[:,i]),
     log_ν(M[:,j])⟩_{g(ν)}` under the metric tensor `g(ν)`.
  3. Solve the simplex-constrained quadratic programme `min_{w≥0, Σw=1} w'Aw`
     using Convex.jl / SCS, which recovers the weights at the Wasserstein
     barycenter.

Optional keyword arguments:
- `N`: number of time steps for each geodesic computation (default 100).
- `tol`: convergence tolerance for each `discrete_transport` call.
- `compute_condition`: if true, prints the condition number of `A`.
- `return_system`: if true, returns `(weights, A)` instead of `weights` alone.
"""
function analysis(ν, M, Q; N=100, tol=1e-10, compute_condition=false, return_system=false)
    p = size(M, 2)

    # compute the tangent vectors of the geodesics from the target measure to each reference
    tangent_vectors = [discrete_transport(Q, ν, M[:,i], N=N, tol=tol).vector.m[1,:,:] for i=1:p]
    g = metric_tensor(ν)

    # form the matrix A for the QP
    A = zeros(p,p)
    for i=1:p, j=i:p
        A[i,j] = A[j,i] = sum(tangent_vectors[i] .* tangent_vectors[j] .* g)
    end

    if compute_condition
        e = abs.(eigvals(A))
        κ = maximum(e) / minimum(e)
        println("Estimated condition number of analysis matrix: $(κ)")
    end
    # solve the QP
    n = size(A, 1)
    x = Variable(n)
    problem = minimize(quadform(x, A))
    # Simplex constraints
    problem.constraints = vcat(problem.constraints, [x >= 0])
    problem.constraints = vcat(problem.constraints, [sum(x) == 1])
    
    Convex.solve!(problem, SCS.Optimizer)
    if return_system
        return (x.value, A)
    end

    x.value  # optimal solution
end
