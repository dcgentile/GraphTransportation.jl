"""
This file contains functionality for solving the projection to K problem, as described in Erbar et al 2020,
section 4.4. In one sense it is perhaps the most involved of all the subroutines, but it does have
a highly convenient aspect to it: like the computation of the proximal mapping of the action conjugate, it
fully decouples index-wise into low dimensional projection problems.

let T be the chosen admissable mean (log mean, geo mean, etc.)
recall that if min(x,y) < 0, T(x,y) = -∞
let K = { (x,y,z) : 0 ≤ z ⩽ T(x,y) }

arguments
let h be the step size, let n be the number of nodes
ρ_minus, ρ_plus, θ ∈ V_{e,h}^0, i.e. tensors of size (1/h) × n × n
"""

function project_K(ρ_m, ρ_p, θ)
    """
    for  each index I, we compute the projection onto K of the point
    p = projection((ρ_minus[I], ρ_plus[I], θ[I]))
    and return three matrices where the first matrix is the composed of the
    x components of p, the second the y components, and the third the z components
    """
    ρ_m_pr, ρ_p_pr, θ_pr = similar(ρ_m), similar(ρ_p), similar(θ)
    for (idx, p) in enumerate(θ)
        p1, p2 = ρ_m[idx], ρ_p[idx]
        ρ_m_pr[idx], ρ_p_pr[idx], θ_pr[idx] = proj_K(p1, p2, p)
    end
    return (ρ_m_pr, ρ_p_pr, θ_pr)
end

function project_K!(ρ_m, ρ_p, θ)
    """
    for  each index I, we compute in place the projection onto K of the point
    p = projection((ρ_minus[I], ρ_plus[I], θ[I])
    I.e, for each index,
    ρ_m[idx], ρ_p[idx], θ[idx] = proj_K(ρ_m[idx], ρ_p[idx], θ[idx])
    """
    @inbounds for idx in eachindex(θ)
        ρ_m[idx], ρ_p[idx], θ[idx] = proj_K(ρ_m[idx], ρ_p[idx], θ[idx])
    end
    return (ρ_m, ρ_p, θ)
end

function proj_K(x, y, z, tolerance=1e-6)
    """
    for real numbers x,y,z, we project to the convex set K
    """
    if 0 ≤ z && z < geomean(x,y)
        """
        if this is true, the point is already in K and we are done
        """
        return (x, y, z)
    elseif z ≤ tolerance
        """
        if (x,y,z) is "underneath" K, this is projection onto the first quadrant of the plane,
        i.e. the bottom facet of K
        """
        return (maximum([x 0]), maximum([y 0]), 0)
    elseif x ≤ 0 && y ≤ 0 && super_differential_inclusion(-x / z, -y / z)
        """
        if z is positive, but both x and y are negative, we may need to project onto the origin
        this happens iff (-x/z, -y/z) ∈ ∂^+T(0) (see Lemma 4.4 for the proof, with details for the
        decision problem itself in Lemma 4.7)
        """
        return (0,0,0)
    else
        """
        finally, if (x,y,z) is above K, concavity of the mean makes the projection problem a smooth
        convex program, which we solve via gradient descent. It is also possible to solve this via Newton's
        method, but then we need some way to guarantee convergence
        """
        return project_by_newton(x,y,z)
        #return project_by_bisection(x,y,z)
    end
end

function super_differential_inclusion(s, t)
    """
    test the point (s,t) for inclustion in the superdifferential at 0
    for the geometric mean, this boils down to the following easy test, and
    can be seen almost immediately by applying lemma 4.6
    """
    return s * t ≥ 0.25 && s > 0 && t > 0
end

function project_by_newton(x,y,z; tolerance=1e-6, safety=false)
    w(q) = [sqrt(q); 1/sqrt(q); 1]
    q_star = find_q([x;y;z])
    w_star = w(q_star)
    τ = ([x;y;z] ⋅ w_star) / (w_star ⋅ w_star)
    p = τ * w_star
    return p

end

function find_q(p)
    # Objective function
    f = q -> begin
        w = [√q, 1/√q, 1]
        n = [-1/(2√q), -√q/2, 1]
        dot(p, w × n)
    end

    # Newton with transformation q = exp(t)
    g = t -> f(exp(t))
    dg = t -> ForwardDiff.derivative(g, t)

    # Try different initial guesses
    initial_guesses = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0]

    for (i, t₀) in enumerate(initial_guesses)
        try
            t_sol = find_zero((g, dg), t₀, Roots.Newton(), atol=1e-10, maxiters=50)
            q = exp(t_sol)

            # Verify solution quality
            if abs(f(q)) < 1e-9
                #i > 1 && @info "Converged with initial guess #$i (t₀=$t₀)"
                return q
            end
        catch
            continue  # Try next initial guess
        end
    end

    # If preset guesses fail, try random initial guesses
    i = 0
    for _ in 1:10000
        i = i+1
        t₀ = randn() * 3  # Random normal, scaled
        try
            t_sol = find_zero((g, dg), t₀, Roots.Newton(), atol=1e-10, maxiters=50)
            q = exp(t_sol)

            if abs(f(q)) < 1e-9
                #@info "Converged with random initial guess (t₀=$t₀)"
                return q
            end
        catch
            continue
        end
    end

    println(i)

    error("Failed to find root after all attempts")
end

# ── Fast alternative: analytic scalar Newton, zero allocations ────────────────

"""
    find_q_fast(x, y, z) -> q

Find q > 0 such that w(q) = [√q, 1/√q, 1] points in the direction of the
projection of (x, y, z) onto the boundary of K.

The projection optimality condition dot(p − τw, dw/dt) = 0 (t = √q) expands,
after multiplying through by 2t², to the quartic

    f(t) = z(1 − t⁴) + t((x − 2y)t² + (2x − y)) = 0

which is solved by a Newton-bisection hybrid with analytic derivative. A bracket
[lo, hi] is established first (f(lo) > 0, f(hi) < 0), then Newton steps are
accepted when they stay inside the bracket and bisection is used otherwise.
No vector allocations, no closures, no external solver library.
"""
function find_q_fast(x, y, z)
    f(t)  = z*(1 - t^4) + t*((x - 2*y)*t^2 + (2*x - y))
    fp(t) = -4*z*t^3 + 3*(x - 2*y)*t^2 + (2*x - y)

    # Establish lo with f(lo) > 0.
    # f(0⁺) = z > 0 always (proj_K_fast only calls us when z > tolerance),
    # so halving from sqrt(z) converges in a few steps regardless of (x,y).
    lo = sqrt(z)
    for _ in 1:60
        f(lo) > 0 && break
        lo /= 2
    end

    # Establish hi with f(hi) < 0.
    hi = max(1.0, (x > 0 && y > 0) ? sqrt(x / y) : 1.0)
    for _ in 1:60
        f(hi) < 0 && break
        hi *= 2
    end

    # If either bracket end is not established (e.g., NaN propagation when hi
    # overflows), fall back to the robust ForwardDiff/Roots solver.
    if !(f(lo) > 0) || !(f(hi) < 0)
        return find_q([x, y, z])
    end

    # Newton-bisection: use Newton step when it stays inside the bracket,
    # otherwise fall back to bisection. Guaranteed to converge.
    t = clamp((x > 0 && y > 0) ? (x / y)^0.25 : 1.0, lo, hi)
    for _ in 1:80
        ft = f(t)
        abs(ft) < 1e-10 && return t^2
        ft > 0 ? (lo = t) : (hi = t)        # tighten bracket
        dft = fp(t)
        t_n = abs(dft) > 1e-14 ? t - ft/dft : (lo + hi)/2
        t   = (lo < t_n < hi) ? t_n : (lo + hi)/2
        hi - lo < 1e-12 && return t^2
    end

    abs(f(t)) < 1e-9 && return t^2
    return find_q([x, y, z])   # final fallback for pathological inputs
end

"""
    project_by_newton_fast(x, y, z) -> (a, b, c)

Allocation-free projection of (x, y, z) onto the boundary of K.
Returns a plain tuple of scalars rather than a Vector.
"""
function project_by_newton_fast(x, y, z)
    q  = find_q_fast(x, y, z)
    sq = sqrt(q)
    w1, w2 = sq, 1/sq
    τ  = (x*w1 + y*w2 + z) / (w1^2 + w2^2 + 1)
    return (τ*w1, τ*w2, τ)
end

function proj_K_fast(x, y, z, tolerance=1e-6)
    gm = sqrt(max(x, 0.0) * max(y, 0.0))
    if 0 ≤ z && z < gm
        return (x, y, z)
    elseif z ≤ tolerance
        return (max(x, 0.0), max(y, 0.0), 0.0)
    elseif x ≤ 0 && y ≤ 0 && super_differential_inclusion(-x / z, -y / z)
        return (0.0, 0.0, 0.0)
    else
        return project_by_newton_fast(x, y, z)
    end
end

function project_K_fast!(ρ_m, ρ_p, θ)
    @inbounds for idx in eachindex(θ)
        ρ_m[idx], ρ_p[idx], θ[idx] = proj_K_fast(ρ_m[idx], ρ_p[idx], θ[idx])
    end
    return (ρ_m, ρ_p, θ)
end
