"""
    BBD(Q::AbstractMatrix, μ::AbstractVector, ν::AbstractVector, N=100; σ=0.5, τ=0.5, maxiters=2^16, tol=1e-10, verbose=false, progress=false)

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
function BBD(Q::AbstractMatrix,
             μ::AbstractVector,
             ν::AbstractVector;
             N=64,
             σ=0.5,
             τ=0.5,
             maxiters=2^16,
             tol=1e-10,
             progress=false,
             )
    a, b = chambolle_pock(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, show_progress=progress)
    return (a, sqrt(action(a)))
end

"""
     BBD(Q::AbstractMatrix,
         steady_state::AbstractVector,
         μ::AbstractVector,
         ν::AbstractVector;
         N=64, σ=0.5, τ=0.5, maxiters=2^16, tol=1e-10, verbose=false, progress=false
     )

REQUIRED ARGS
Q: Markov transition rate matrix representing the underlying graph
steady_state: precomputed steady-state distribution of Q
μ: probability measure w.r.t. the steady state of Q
ν: probability measure w.r.t. the steady state of Q
initial guess: provide an ErbarVector containing an "initial guess" for the geodesic to be computed

OPTIONAL ARGS:
N: integer, determines how many steps are used for computing the geodesics which yield the tangent vector
σ: positive float, must satisfy στ < 1, roughly the equivalent of a learning rate
τ: positive float, must satisfy στ < 1, roughly the equivalent of a learning rate
λ: float in (0,1)
tol: positive float, convergence threshold
maxiters: positive integer, cap on number of iterations for scheme
verbose: boolean, DO NOT USE WITH PROGRESS if true, causes Chambolle Pock routine to print status everytime it passes to a new step of the computation.
progress: boolean, shows a progress meter if true
"""


function BBD(Q::AbstractMatrix,
             steady_state::AbstractVector,
             μ::AbstractVector,
             ν::AbstractVector;
             N=64,
             σ=0.5,
             τ=0.5,
             maxiters=2^16,
             tol=1e-10,
             progress=false,
             )
    a, b = chambolle_pock(Q, steady_state, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, show_progress=progress)
    return (a, sqrt(action(a)))
end

"""
    BBD(Q, μ, ν, initial_guess)

REQUIRED ARGS
Q: Markov transition rate matrix representing the underlying graph
μ: probability measure w.r.t. the steady state of Q
ν: probability measure w.r.t. the steady state of Q
initial guess: provide an ErbarVector containing an "initial guess" for the geodesic to be computed

OPTIONAL ARGS:
N: integer, determines how many steps are used for computing the geodesics which yield the tangent vector
σ: positive float, must satisfy στ < 1, roughly the equivalent of a learning rate
τ: positive float, must satisfy στ < 1, roughly the equivalent of a learning rate
λ: float in (0,1)
tol: positive float, convergence threshold
maxiters: positive integer, cap on number of iterations for scheme
verbose: boolean, DO NOT USE WITH PROGRESS if true, causes Chambolle Pock routine to print status everytime it passes to a new step of the computation.
progress: boolean, shows a progress meter if true
"""

function BBD(
    Q::AbstractMatrix,
    μ::AbstractVector,
    ν::AbstractVector,
    initial_guess::ErbarVector;
    N=64,
    tol=1e-10,
    progress=false,
    maxiters=2^16,
    σ=0.5,
    τ=0.5,
    λ=1.0
)
    new_cache = ErbarCache(Q, μ, ν, N)
    new_bundle = ErbarBundle(new_cache, initial_guess)
    geodesic, _ = chambolle_pock(new_bundle, maxiters=maxiters, tol=tol, σ=σ, τ=τ, λ=λ, show_progress=progress)
    return (geodesic, sqrt(action(geodesic)))
end

