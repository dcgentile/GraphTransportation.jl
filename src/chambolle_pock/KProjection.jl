"""
    project_K(ρ_m, ρ_p, θ) -> (ρ_m_new, ρ_p_new, θ_new)

Project each element of the triple `(ρ_m[I], ρ_p[I], θ[I])` onto the convex
set `K = {(x,y,z) : 0 ≤ z ≤ T(x,y)}` (where `T` is the geometric mean) and
return new arrays.  All three input tensors must have the same shape.

See Erbar et al. 2020 section 4.4.
"""
function project_K(ρ_m, ρ_p, θ)
    ρ_m_pr, ρ_p_pr, θ_pr = similar(ρ_m), similar(ρ_p), similar(θ)
    for (idx, p) in enumerate(θ)
        p1, p2 = ρ_m[idx], ρ_p[idx]
        ρ_m_pr[idx], ρ_p_pr[idx], θ_pr[idx] = proj_K(p1, p2, p)
    end
    return (ρ_m_pr, ρ_p_pr, θ_pr)
end

"""
    project_K!(ρ_m, ρ_p, θ) -> (ρ_m, ρ_p, θ)

In-place variant of `project_K`: projects each element-wise triple onto K and
writes the result back into the input arrays.
"""
function project_K!(ρ_m, ρ_p, θ)
    @inbounds for idx in eachindex(θ)
        ρ_m[idx], ρ_p[idx], θ[idx] = proj_K(ρ_m[idx], ρ_p[idx], θ[idx])
    end
    return (ρ_m, ρ_p, θ)
end

"""
    proj_K(x, y, z, tolerance=1e-6) -> (a, b, c)

Project the scalar triple `(x, y, z)` onto `K = {(x,y,z) : 0 ≤ z ≤ √(xy)}`.

Four cases are handled in order:
1. Already in K → return unchanged.
2. Below K (`z ≤ tolerance`) → project onto the bottom facet (non-negative orthant of the `z=0` plane).
3. Both `x,y ≤ 0` and `(-x/z, -y/z) ∈ ∂⁺T(0)` (Lemma 4.7) → project to origin.
4. Above K → solve the smooth convex projection via `project_by_newton`.
"""
function proj_K(x, y, z, tolerance=1e-6)
    if 0 ≤ z && z < geomean(x,y)
        return (x, y, z)
    elseif z ≤ tolerance
        return (maximum([x 0]), maximum([y 0]), 0)
    elseif x ≤ 0 && y ≤ 0 && super_differential_inclusion(-x / z, -y / z)
        return (0,0,0)
    else
        return project_by_newton(x,y,z)
    end
end

"""
    super_differential_inclusion(s, t) -> Bool

Return `true` if `(s, t)` lies in the superdifferential of the geometric mean
at `0`, i.e. `s·t ≥ 1/4` with `s > 0` and `t > 0`.

Used by `proj_K` to decide whether to project to the origin (Lemma 4.6–4.7 of
Erbar et al. 2020).
"""
function super_differential_inclusion(s, t)
    return s * t ≥ 0.25 && s > 0 && t > 0
end

"""
    project_by_newton(x, y, z; tolerance=1e-6, safety=false) -> (a, b, c)

Project `(x, y, z)` onto the boundary of K by solving for the scalar `q > 0`
such that `w(q) = [√q, 1/√q, 1]` points in the direction of the projection,
using the `find_q` solver (ForwardDiff + Roots).

This is the reference implementation; the production path uses
`project_by_newton_fast`, which avoids allocations and external solver calls.
"""
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

"""
    proj_K_fast(x, y, z, tolerance=1e-6) -> (a, b, c)

Allocation-free scalar projection onto `K = {(x,y,z) : 0 ≤ z ≤ √(xy)}`.
Mirrors the case logic of `proj_K` but uses `project_by_newton_fast` for the
boundary case, avoiding all heap allocations.  This is the production path
called by `project_K_fast!` in the Chambolle-Pock inner loop.
"""
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

"""
    project_K_fast!(ρ_m, ρ_p, θ) -> (ρ_m, ρ_p, θ)

In-place variant of `proj_K_fast` applied element-wise to all index triples.
The production entry point used by `prox_G!` in the Chambolle-Pock hot loop.
"""
function project_K_fast!(ρ_m, ρ_p, θ)
    @inbounds for idx in eachindex(θ)
        ρ_m[idx], ρ_p[idx], θ[idx] = proj_K_fast(ρ_m[idx], ρ_p[idx], θ[idx])
    end
    return (ρ_m, ρ_p, θ)
end
