using Base.Threads

"""
this file contains the Chambolle pock routine for minimizing the objective function described in eqn (26)
of Erbar et al 2020.

RELEVANT SETS:

Javg = { ρ, ρ_bar : ρ_bar == avg_h(ρ) } (avg_h is the time average operator, see definition 3.2 in Erbar 2020)
Jeq = { ρ_bar, q : ρ_bar == q }
Jpm = { q, ρ_minus, ρ_plus : ρ_minus[t,i,j] == q[t,i] && ρ_plus[t,i,j] == q[t,j]}
K = { x,y,z : 0 ≤ z ≤ T(x,y) } (here T is the chosen admissible mean)
ScriptK = { ρ_minus, ρ_plus, θ: ρ_minus[t,i,j], ρ_plus[t,i,j], θ[t,i,j] ∈ K}
"""

"""
    chambolle_pock_routine(a::ErbarBundle, b::ErbarBundle, a_bar::ErbarBundle, a_next::ErbarBundle, b_next::ErbarBundle, a_bar_next::ErbarBundle, c::ErbarBundle, d::ErbarBundle; σ=0.5, τ=0.5, λ=1.0, maxiters=2^16, tol=1e-10, show_progress=true)

Description of the function.

#TODO
"""
function chambolle_pock_routine(
    a::ErbarBundle,
    b::ErbarBundle,
    a_bar::ErbarBundle,
    a_next::ErbarBundle,
    b_next::ErbarBundle,
    a_bar_next::ErbarBundle,
    c::ErbarBundle,
    d::ErbarBundle;
    σ=0.5,
    τ=0.5,
    λ=1.0,
    maxiters=2^16,
    tol=1e-10,
    show_progress=true,
    )
    show_progress ? p = ProgressUnknown(spinner=true) : 0
    normdiff = Inf
    N = a.cache.N

    for i in 1:maxiters
        show_progress ? next!(p;
                              showvalues=[
                                  ("Difference in norm between iterations", normdiff),
                                  ("Current iteration", i),
                                  ("Number of steps", N),
                                  ("Convergence tolerance", tol),
                                  ("σ", σ),
                                  ("τ", τ)
                              ]) : nothing
        combine!(c, b, a_bar, 1.0, σ)
        prox_Fstar!(b_next, c)
        combine!(c, a, b_next, 1.0, -τ)
        prox_G!(a_next, c)
        combine!(d, a_next, a, 1.0, -1.0)
        normdiff = sum(d.vector.ρ .* d.vector.ρ * d.cache.π)
        if normdiff < tol
            return (a_next, b_next)
        end
        λ = 1 / √(1 + 2 * τ)
        τ *= λ
        σ /= λ

        combine!(a_bar_next, a_next, d, 1.0, λ)
        assign!(a, a_next)
        assign!(b, b_next)
        assign!(a_bar, a_bar_next)
    end
    @warn "Chambolle Pock did not converge in $(maxiters) steps. Last recorded norm difference: $(normdiff)"
    return (a, b)
end


# a more memory efficient version of ChamPock
"""
    chambolle_pock(a::ErbarBundle;maxiters=2^16, tol=1e-10, σ=0.5, τ=0.5, λ=1.0, show_progress=false)

Description of the function.

#TODO
"""
function chambolle_pock(a::ErbarBundle;maxiters=2^16, tol=1e-10, σ=0.5, τ=0.5, λ=1.0, show_progress=false)
    b = copy(a)
    a_bar = copy(a)
    a_next = copy(a)
    b_next = copy(a)
    a_bar_next = copy(a)
    c = copy(a)
    d = copy(a)
    return chambolle_pock_routine(a, b, a_bar, a_next, b_next, a_bar_next, c, d, σ=σ, τ=τ, λ=λ, maxiters=maxiters, tol=tol, show_progress=show_progress)
end


function chambolle_pock(
    Q::AbstractMatrix,
    μ::AbstractVector,
    ν::AbstractVector,
    N::Int;
    maxiters=2^16,
    σ=0.5,
    τ=0.5,
    λ=1.0,
    tol=1e-10,
    show_progress=false
)
    """
    this is a memory efficient version of Chambolle Pock that does computations in place whenever possible
    iterations cease if ∫||ρ_{k} - ρ_{k + 1}||_π dt < tol

    arguments
    Q, a Markov kernel defining the graph
    μ, ν, probability densities w.r.t to the steady state π of Q
    N, the number of steps in the geodesic
    """
    # we will only ever use 8 vectors
    a = ErbarBundle(Q, μ, ν, N)
    b = ErbarBundle(Q, μ, ν, N)
    a_bar = ErbarBundle(Q, μ, ν, N)
    a_next = ErbarBundle(Q, μ, ν, N)
    b_next = ErbarBundle(Q, μ, ν, N)
    a_bar_next = ErbarBundle(Q, μ, ν, N)
    c = ErbarBundle(Q, μ, ν, N)
    d = ErbarBundle(Q, μ, ν, N)
    return chambolle_pock_routine(
        a, b, a_bar, a_next, b_next, a_bar_next, c, d,
        maxiters=maxiters, tol=tol, σ=σ, τ=τ, λ=λ)
end


"""
#TODO Describe function
"""
function chambolle_pock(
    Q::AbstractMatrix,
    steady_state::AbstractVector,
    μ::AbstractVector,
    ν::AbstractVector,
    N::Int;
    maxiters=2^16,
    σ=0.5,
    τ=0.5,
    λ=1.0,
    tol=1e-10,
    show_progress=false
)
    # we will only ever use 8 vectors
    a = ErbarBundle(Q, steady_state, μ, ν, N)
    b = ErbarBundle(Q, steady_state, μ, ν, N)
    a_bar = ErbarBundle(Q, steady_state, μ, ν, N)
    a_next = ErbarBundle(Q, steady_state, μ, ν, N)
    b_next = ErbarBundle(Q, steady_state, μ, ν, N)
    a_bar_next = ErbarBundle(Q, steady_state, μ, ν, N)
    c = ErbarBundle(Q, steady_state, μ, ν, N)
    d = ErbarBundle(Q, steady_state, μ, ν, N)
    return chambolle_pock_routine(
        a, b, a_bar, a_next, b_next, a_bar_next, c, d,
        maxiters=maxiters, tol=tol, σ=σ, τ=τ, λ=λ)
end

"""
    prox_Fstar!(targ, bundle)

compute the proximal mapping of F star in place
this amounts to computing the proximal mappings of the conjuage Action, IJPM, and IJAvg
"""

function prox_Fstar!(targ, bundle)
    cache = bundle.cache
    v = bundle.vector
    u = targ.vector
    @threads for task_id in 1:3
        if task_id == 1
            v.θ, v.m = prox_Astar!(v.θ, v.m)
        elseif task_id == 2
            v.q, v.ρ_minus, v.ρ_plus = proximal_IJpm_star!(v.q, v.ρ_minus, v.ρ_plus, cache.Q)
        elseif task_id == 3
            v.ρ, v.ρ_avg = prox_IJavg_star!(v.ρ, v.ρ_avg, cache.μ, cache.ν, cache.avg_sys)
        end
    end
    u.ρ .= v.ρ
    u.θ .= v.θ
    u.m .= v.m
    u.ρ_minus .= v.ρ_minus
    u.ρ_plus .= v.ρ_plus
    u.ρ_avg .= v.ρ_avg
    u.q .= v.q
end

"""
    prox_G!(targ, bundle)

compute the proximal mapping of G
this amounts to computing the projection to the space of solutions to the Galerkin-discretized continuity equation,
projection to the set Script K, and projection to the set Jeq
"""

function prox_G!(targ, bundle)
    cache = bundle.cache
    v = bundle.vector
    u = targ.vector
    for task_id in 1:3
        if task_id == 1
            v.ρ, v.m = proj_CE!(v.ρ, v.m, cache.μ, cache.ν, cache.Q, cache.ceh_sys)
        elseif task_id == 2
            v.ρ_minus, v.ρ_plus, v.θ = project_K!(v.ρ_minus, v.ρ_plus, v.θ)
        elseif task_id == 3
            v.ρ_avg, v.q = project_IJeq!(v.ρ_avg, v.q)
        end

    end


    u.ρ .= v.ρ
    u.θ .= v.θ
    u.m .= v.m
    u.ρ_minus .= v.ρ_minus
    u.ρ_plus .= v.ρ_plus
    u.ρ_avg .= v.ρ_avg
    u.q .= v.q
end
