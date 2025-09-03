#using LinearAlgebra
#include("utils.jl")
#include("ErbarVector.jl")
#include("galerkin/Chambolle.jl")

function BBD(Q::AbstractMatrix,
             μ::AbstractVector,
             ν::AbstractVector,
             N=100;
             σ=0.5,
             τ=0.5,
             maxiters=2^16,
             tol=1e-10,
             verbose=false
             )
    geodesic = chambolle_pock(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, verbose=verbose)
    return (geodesic, sqrt(action(geodesic)))
end

function BBD(Q, μ, ν, N, initial_guess)
    new_cache = ErbarCache(Q, μ, ν, N)
    new_bundle = ErbarBundle(new_cache, initial_guess)
    geodesic = chambolle_pock(new_bundle)
    return (geodesic, sqrt(action(geodesic)))
end

function variance(Q, ν, M, λ, N)
    #M is a P x N matrix, where each row corresponds to a measure
    v = 0
    for (idx, row) in enumerate(eachrow(M))
        v += 0.5 * λ[idx] * action(chambolle_pock(Q, ν, row, N))
    end
    println(v)
    return v
end

function accelerated_descent(Q, M, λ, N; maxiters=100, h=1e-1, tol=1e-10)
    V,  = size(Q)
    u = steady_state_from_adjacency(Q)
    x_prev = ones(V-1) # initial guess: uniform distribution
    x_curr = ones(V-1) # initial guess: uniform distribution
    x_next = ones(V-1)

    J = variance_functional(Q, M, λ, N)
    F(x) = J(surface_point(x, u))
    α = 0.5
    f0 = 0
    f1 = 0
    β = 0
    η = 0
    diff = Inf

    p = ProgressUnknown(spinner=true)
    for i in 1:maxiters
        next!(p; showvalues=[("Descent Iteration: ", i), ("Objective Step Diff: ", diff)])
        # approximate the gradient
        y = x_curr + β * (x_curr - x_prev)
        fy = F(y)

        for j in 1:V-1
            ∂jf = finite_difference(y, j, h, u, fy, F)
            x_next[j] = y[j] - α * ∂jf
        end
        η_next = 0.5 * (1 + √(1 + 4*η^2))
        β = (η - 1) / η_next
        η = η_next

        # have we converged?
        f1 = F(x_next)
        diff = abs(f1 - f0)
        if diff < tol
            return (surface_point(x_next, u), f1)
        end
        x_prev = x_curr
        x_curr = x_next
    end

    @warn "Did not converge in $(maxiters) iterations"
    return (surface_point(x_next, u), f1)
end

function descent(Q, M, λ, N; maxiters=100, h=1e-2, tol=1e-10)
    V,  = size(Q)
    u = steady_state_from_adjacency(Q)
    x0 = ones(V-1) # initial guess: uniform distribution
    x1 = zeros(V-1)

    J = variance_functional(Q, M, λ, N)
    F(x) = J(surface_point(x, u))
    f0 = F(x0)
    f1 = 0
    p = ProgressUnknown(spinner=true)

    for i in 1:maxiters
        next!(p; showvalues=[("Descent Iteration: ", i)])
        # approximate the gradient
        for j in 1:V-1
            ∂jf = finite_difference(x0, j, h, u, f0, F)
            x1[j] = x0[j] - i^-2 * ∂jf
        end
        # have we converged?
        f1 = F(x1)
        if abs(f1 - f0) < tol
            return (surface_point(x1, u), f1)
        end
        x0, f0 = x1, f1
    end

    @warn "Did not converge in $(maxiters) iterations"
    return (surface_point(x1, u), f1)
end

function finite_difference(x::AbstractVector, j::Integer, h::Float64, u::AbstractVector, f0::Float64, F::Function)
    N = size(x, 1)
    v = h * e(j,N)
    if 0 < x[j] - h && x[j] + h < (1 / u[j])
        # middle difference
        return (F(x + v) - F(x - v)) / (2*h)
    elseif 0 < x[j] - h && x[j] + h > (1 / u[j])
        #left difference
        return (F(x - v) - f0) / h
    elseif x[j] - h < 0 && x[j] + h < (1 / u[j])
        #right  difference
        return (F(x + v) - f0) / h
    else
        error("Invalid step error")
    end

end


function variance_functional(Q, M, λ, N)
	return f(ν) = variance(Q, ν, M, λ, N)
end

coordinate_chart(p) = p[1:end-1]
surface_point(x, u) = vcat(x, [1 - sum(x .* u[1:end-1])] / u[end])
e(i,n) = I[1:n, i]

function simplex_points(Q)
	u = steady_state_from_adjacency(Q)
    id = I(size(u,1))
    return (.!id ./ (1 .- u))[:,1:end-1]
end
