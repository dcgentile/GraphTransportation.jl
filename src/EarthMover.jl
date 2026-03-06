"""
    discrete_transport(Q::AbstractMatrix, μ::AbstractVector, ν::AbstractVector, N=100; σ=0.5, τ=0.5, maxiters=2^16, tol=1e-10, verbose=false, progress=false)

REQUIRED ARGS
Q: Markov transition rate matrix representing the underlying graph
μ: probability measure w.r.t. the steady state of Q
ν: probability measure w.r.t. the steady state of Q
weights: a non-negative vector of size num_measures with sum(weights) == 1

OPTIONAL ARGS:
N: integer, determines how many steps are used for computing the geodesics which yield the tangent vector
σ: positive float, must satisfy στ < 1, roughly the equivalent of a learning rate
τ: positive float, must satisfy στ < 1, roughly the equivalent of a learning rate
tol: positive float, convergence threshold
maxiters: positive integer, cap on number of iterations for scheme
verbose: boolean, DO NOT USE WITH PROGRESS if true, causes Chambolle Pock routine to print status everytime it passes to a new step of the computation.
progress: boolean, shows a progress meter if true
"""
function discrete_transport(
    Q::AbstractMatrix,
    μ::AbstractVector,
    ν::AbstractVector;
    N=64,
    σ=0.5,
    τ=0.5,
    maxiters=2^16,
    tol=1e-10,
    progress=false,
    initialization=nothing,
)
    if isnothing(initialization)
        a = chambolle_pock(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, show_progress=progress)
    else
        # Rebind the previous result to the new boundary conditions (μ, ν),
        # reusing the cached linear systems since they only depend on Q and N.
        new_cache = ErbarCache(initialization.cache, μ, ν)
        warm = ErbarBundle(new_cache, copy(initialization.vector))
        a = chambolle_pock(warm, maxiters=maxiters, σ=σ, τ=τ, tol=tol, show_progress=progress)
    end
    return a
end


"""
    transport_cost(Q::AbstractMatrix, μ::AbstractVector, ν::AbstractVector, N=100; σ=0.5, τ=0.5, maxiters=2^16, tol=1e-10, verbose=false, progress=false)

REQUIRED ARGS
Q: Markov transition rate matrix representing the underlying graph
μ: probability measure w.r.t. the steady state of Q
ν: probability measure w.r.t. the steady state of Q
weights: a non-negative vector of size num_measures with sum(weights) == 1

OPTIONAL ARGS:
N: integer, determines how many steps are used for computing the geodesics which yield the tangent vector
σ: positive float, must satisfy στ < 1, roughly the equivalent of a learning rate
τ: positive float, must satisfy στ < 1, roughly the equivalent of a learning rate
tol: positive float, convergence threshold
maxiters: positive integer, cap on number of iterations for scheme
verbose: boolean, DO NOT USE WITH PROGRESS if true, causes Chambolle Pock routine to print status everytime it passes to a new step of the computation.
progress: boolean, shows a progress meter if true
"""
function transport_cost(Q::AbstractMatrix,
             μ::AbstractVector,
             ν::AbstractVector;
             N=64,
             σ=0.5,
             τ=0.5,
             maxiters=2^16,
             tol=1e-10,
             progress=false,
             )
    a = chambolle_pock(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, show_progress=progress)
    return sqrt(action(a))
end
