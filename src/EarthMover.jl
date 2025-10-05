"""
    BBD(Q::AbstractMatrix, μ::AbstractVector, ν::AbstractVector, N=100; σ=0.5, τ=0.5, maxiters=2^16, tol=1e-10, verbose=false, progress=false)

Description of the function.

#TODO
"""
function BBD(Q::AbstractMatrix,
             μ::AbstractVector,
             ν::AbstractVector;
             N=64,
             σ=0.5,
             τ=0.5,
             maxiters=2^16,
             tol=1e-10,
             verbose=false,
             progress=false,
             )
    a, b = chambolle_pock(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, verbose=verbose, show_progress=progress)
    return (a, sqrt(action(a)))
end

function BBD(Q::AbstractMatrix,
             steady_state::AbstractVector,
             μ::AbstractVector,
             ν::AbstractVector;
             N=64,
             σ=0.5,
             τ=0.5,
             maxiters=2^16,
             tol=1e-10,
             verbose=false,
             progress=false,
             )
    a, b = chambolle_pock(Q, steady_state, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, verbose=verbose, show_progress=progress)
    return (a, sqrt(action(a)))
end

"""
    BBD(Q, μ, ν, initial_guess)

Description of the function.

#TODO
"""
function BBD(
    Q::AbstractMatrix,
    μ::AbstractVector,
    ν::AbstractVector,
    initial_guess::ErbarVector;
    N=64,
    tol=1e-10,
    verbose=false,
    progress=false,
    maxiters=2^16,
    σ=0.5,
    τ=0.5,
    λ=1.0
)
    new_cache = ErbarCache(Q, μ, ν, N)
    new_bundle = ErbarBundle(new_cache, initial_guess)
    geodesic = chambolle_pock(new_bundle, maxiters=maxiters, verbose=verbose, tol=tol, σ=σ, τ=τ, λ=λ, show_progress=progress)
    return (geodesic, sqrt(action(geodesic)))
end

