"""
    discrete_transport(Q, μ, ν; N=64, σ=0.5, τ=0.5, maxiters=2^16, tol=1e-10,
                       progress=false, initialization=nothing) -> ErbarBundle

Compute the discrete transport geodesic from `μ` to `ν` on the graph defined
by the Markov transition matrix `Q`, using the Chambolle-Pock primal-dual
algorithm of Erbar et al. 2020.

Returns an `ErbarBundle` encoding the full geodesic; its `vector.m[1,:,:]`
field is the initial tangent vector (logarithmic map at `μ`), and `action(result)`
gives the squared discrete transport distance.

# Arguments
- `Q`: row-stochastic Markov transition matrix defining the graph
- `μ`, `ν`: probability densities w.r.t. the stationary distribution of `Q`
- `N`: number of time steps for the geodesic discretisation (default 64)
- `σ`, `τ`: Chambolle-Pock step sizes; must satisfy `σ·τ < 1` (default 0.5)
- `maxiters`: maximum Chambolle-Pock iterations (default 65536)
- `tol`: convergence tolerance on the density change between iterates
- `progress`: show a progress spinner if `true`
- `initialization`: an `ErbarBundle` from a prior solve to warm-start from
- `adaptive`: use Chambolle-Pock's accelerated (strongly-convex) step-size schedule.
  Default `false`; this problem has no strongly convex term, so `true` converges to a
  biased wrong answer on graphs with more than 2 nodes (see `chambolle_pock_routine`).
  Kept only to reproduce results computed before this was discovered.
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
    adaptive=false,
)
    if isnothing(initialization)
        a = chambolle_pock(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, show_progress=progress, adaptive=adaptive)
    else
        # Rebind the previous result to the new boundary conditions (μ, ν),
        # reusing the cached linear systems since they only depend on Q and N.
        new_cache = ErbarCache(initialization.cache, μ, ν)
        warm = ErbarBundle(new_cache, copy(initialization.vector))
        a = chambolle_pock(warm, maxiters=maxiters, σ=σ, τ=τ, tol=tol, show_progress=progress, adaptive=adaptive)
    end
    return a
end


"""
    transport_cost(Q, μ, ν; N=64, σ=0.5, τ=0.5, maxiters=2^16, tol=1e-10,
                   progress=false) -> Float64

Return the discrete transport distance `W(μ, ν)` on the graph defined by `Q`,
computed as `√(action(discrete_transport(Q, μ, ν; ...)))`.

# Arguments
- `Q`: row-stochastic Markov transition matrix defining the graph
- `μ`, `ν`: probability densities w.r.t. the stationary distribution of `Q`
- `N`: number of time steps for the geodesic discretisation (default 64)
- `σ`, `τ`: Chambolle-Pock step sizes; must satisfy `σ·τ < 1` (default 0.5)
- `maxiters`: maximum Chambolle-Pock iterations (default 65536)
- `tol`: convergence tolerance on the density change between iterates
- `progress`: show a progress spinner if `true`
- `adaptive`: see `discrete_transport`; default `false` (the accelerated schedule is
  biased for this problem on graphs with more than 2 nodes).
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
             adaptive=false,
             )
    a = chambolle_pock(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, show_progress=progress, adaptive=adaptive)
    return sqrt(action(a))
end
