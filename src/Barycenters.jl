"""
    step_direction(ν, M, weights, Q; tol=1e-10, n_steps=100,
                   prev_geodesics=nothing) -> (tangent_vector, variance, geodesics)

Compute the discrete transport gradient direction and variance at `ν` for the
weighted Fréchet mean objective with reference measures `M` and `weights`.

For each reference measure `M[:,i]`, computes the geodesic from `ν` to
`M[:,i]` via `discrete_transport` and accumulates the weighted sum of initial
tangent vectors.

# Arguments
- `ν`: current iterate — probability density w.r.t. the stationary distribution of `Q`
- `M`: `V × p` matrix whose columns are the reference probability densities
- `weights`: non-negative length-`p` vector summing to 1
- `Q`: row-stochastic Markov transition matrix defining the graph
- `tol`: convergence tolerance for each `discrete_transport` call
- `n_steps`: number of geodesic time steps
- `prev_geodesics`: vector of `ErbarBundle`s from the previous iteration for warm-starting

# Returns
`(tangent_vector, variance, geodesics)` where `tangent_vector` is the gradient
direction, `variance` is the current objective value, and `geodesics` is the
vector of solved `ErbarBundle`s (for warm-starting the next call).
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
    barycenter(M, weights, Q; h=1.0, maxiters=100, tol=1e-8,
               geodesic_tol=1e-10, geodesic_steps=100,
               return_stats=false, initialization_index=1,
               initialization=nothing, uniform_init=false,
               geodesic_warmstart=true, verbose=true,
               stagnation_window=100, stagnation_tol=0.01,
               h_min=1e-8) -> ν  (or (ν, norm_diffs, variances) if return_stats=true)

Compute the discrete transport barycenter of the reference measures in `M` with
respect to `weights` via gradient descent on the graph defined by `Q`.

Convergence is assessed on the gradient norm `‖∇J‖`; a stagnation detector
halves the step size when relative improvement over the last `stagnation_window`
iterates falls below `stagnation_tol`.

# Arguments
- `M`: `V × p` matrix whose columns are reference probability densities
- `weights`: non-negative length-`p` vector summing to 1
- `Q`: row-stochastic Markov transition matrix defining the graph
- `h`: initial WGD step size (default 1.0)
- `maxiters`: maximum number of WGD iterations
- `tol`: gradient-norm convergence threshold
- `geodesic_tol`: convergence tolerance passed to each `discrete_transport` call
- `geodesic_steps`: number of time steps for each geodesic
- `return_stats`: if `true`, return `(ν, norm_diffs, variances)` instead of `ν`
- `initialization`: explicit starting measure; takes precedence over index-based init
- `initialization_index`: column of `M` used as the starting point (default 1)
- `uniform_init`: if `true`, start from the uniform density (density ≡ 1)
- `geodesic_warmstart`: reuse geodesics from the previous iteration as warm starts
- `verbose`: show a ProgressMeter display with per-iteration statistics
- `stagnation_window`: number of iterations over which to measure relative improvement
- `stagnation_tol`: minimum relative improvement before halving `h`
- `h_min`: minimum allowed step size; stagnation halving stops here
"""
function barycenter(M, weights, Q;
                    h=1., maxiters=100, tol=1e-8,
                    geodesic_tol=1e-10, geodesic_steps=100,
                    return_stats=false, initialization_index=1,
                    initialization=nothing, uniform_init=false,
                    geodesic_warmstart=true, verbose=true,
                    stagnation_window=100, stagnation_tol=0.01, h_min=1e-8)
    ν = if !isnothing(initialization)
        copy(initialization)
    elseif uniform_init
        ones(size(M, 1))   # density ≡ 1 w.r.t. stationary measure — full support everywhere
    else
        M[:, initialization_index]
    end
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
        # Gradient norm is h-independent; used for convergence so that stagnation-driven
        # step size reductions cannot trigger false convergence.
        grad_norm = norm(div_term .* root_steady_state)

        ν_next = ν .- h * div_term

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

        if k > stagnation_window
            relative_improvement = (norm_diffs[k - stagnation_window] - norm_diff) / norm_diff
            if relative_improvement < stagnation_tol
                if h <= h_min
                    verbose && ProgressMeter.finish!(prog; desc="WGD (stagnated, h_min reached) ")
                    break
                else
                    h /= 2
                    verbose && println("\nStagnation detected — halving step size to h=$(h)")
                end
            end
        end

        if verbose
            ProgressMeter.next!(prog; showvalues=[
                ("‖∇J‖", @sprintf("%.4e", grad_norm)),
                ("‖Δν‖", @sprintf("%.4e", norm_diff)),
                ("var",  @sprintf("%.4e", variance)),
                ("h",    @sprintf("%.4e", h)),
            ])
        end

        if grad_norm < tol
            verbose && ProgressMeter.finish!(prog; desc="WGD (converged) ")
            break
        else
            ν = ν_next
        end

    end
    return return_stats ? (ν_next, norm_diffs, variances) : ν_next
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
     using Convex.jl / SCS, which recovers the weights at the discrete transport
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
