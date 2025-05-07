include("ErbarVector.jl")
include("galerkin/Chambolle.jl")
include("utils.jl")

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
    return v
end

function descent(Q, M, λ, N; maxiters=100, h=1e-3)
    V,  = size(Q)
    u = steady_state_from_adjacency(Q)
    x0 = ones(V-1) # initial guess: uniform distribution
    x1 = zeros(V-1)
    f0 = variance(x0)
    f1 = 0

    for i in 1:maxiters
        # approximate the gradient
        for j in V
            ∂jf = finite_difference(x0, j, h, f0, u)
            x1[j] = x0[j] - i^-2 * ∂jf
        end
        # have we converged?
        f1 = F(x1)
        if abs(f1 - f0) < tol
            return (x1, f1)
        end
    end

    @warn "Did not converge in $(maxiters) iterations"
    return (x1, f1)
end

function finite_difference(x, j, h, u, f0)
    if 0 < x[j] - h && x[j] + h < 1 / u[j]
        # middle difference
        return
    elseif 0 < x[j] - h && x[j] + h > 1 / u[j]
        #left difference
        return
    elseif x[j] - h < 0 && x[j] + h < 1 / u[j]
        #right  difference
        return
    else
        raise("Invalid step error")
    end

end

coordinate_chart(p) = p[1:end-1]
surface_point(x, u) = vcat(x, [1 - sum(x .* u[1:end-1])] / u[end])
F(x) = variance(surface_point(x))
