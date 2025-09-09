function BBD(Q::AbstractMatrix,
             μ::AbstractVector,
             ν::AbstractVector,
             N=100;
             σ=0.5,
             τ=0.5,
             maxiters=2^16,
             tol=1e-10,
             verbose=false,
             progress=false,
             )
    geodesic = chambolle_pock(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, verbose=verbose, show_progress=progress)
    return (geodesic, sqrt(action(geodesic)))
end

function BBD(Q, μ, ν, N, initial_guess)
    new_cache = ErbarCache(Q, μ, ν, N)
    new_bundle = ErbarBundle(new_cache, initial_guess)
    geodesic = chambolle_pock(new_bundle)
    return (geodesic, sqrt(action(geodesic)))
end

