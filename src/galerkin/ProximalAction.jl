"""
This file contains functionality for solving the projection to B problem, as described in Erbar et al 2020,
section 4.3.

Let B = { (p,q) : p + 0.25 * q^2 ≤ 0 }
given θ, m, to compute the proximal mapping of the conjugate of the edgewise action A, we must solve indexwise the projection
onto B. This is accomplished via Newton's method.

arguments
let h be the step size, let n be the number of nodes
θ,m ∈ V_{e,h}^0, i.e. tensors of size (1/h) × n × n
"""
function prox_Astar!(θ::AbstractArray, m::AbstractArray)
    """
    solve the indexwise projection problems in place

    arguments
    θ, m ∈ V_{e,h}^0, i.e. they are tensors of size (1/h) × n × n, where h is the step size and n the number of nodes
    """
    #θ, m = proj_B.(θ, m)
    for i in eachindex(θ, m)
        #θ[i], m[i] = proj_B(θ[i], m[i])
        #θ[i], m[i] = project_by_bisection(θ[i], m[i])
        θ[i], m[i] = projection_by_newton(θ[i], m[i])
    end
    return (θ, m)
end

function prox_Astar(θ, m)
    """
    solve the indexwise projection and return a new pair of arrays

    arguments
    θ, m ∈ V_{e,h}^0, i.e. they are tensors of size (1/h) × n × n, where h is the step size and n the number of nodes
    """
	θ_pr, m_pr = similar(θ), similar(m)
    for idx in eachindex(θ)
        θ_pr[idx], m_pr[idx] = project_by_bisection(θ[idx], m[idx])
    end

    return (θ_pr, m_pr)
end


"""
    project_by_bisection(a, b; tol=1e-5, maxiters=2^16) -> (p, q)

Project the point `(a, b)` onto the set `B = {(p, q) : p + 0.25·q² ≤ 0}` using
bisection on the scalar optimality condition.

Points already inside `B` are returned unchanged.  For exterior points the
projection lies on the parabolic boundary `p = -0.25·q²`; the signed magnitude
`|q|` is found by bracketing and bisecting the monotone residual function.

This is kept as a reference implementation.  The production path uses
`projection_by_newton`, which solves the same problem analytically via
Cardano's formula and is allocation-free.
"""
function project_by_bisection(a,b; tol=1e-5, maxiters=2^16)
    if a + 0.25 * b^2 ≤ 0
        return (a,b)
    end

    s = sign(b)
    bhat = s * b
    u = bhat
    l = 0

    for _ in 1:maxiters
        t0 = 0.5 * (u + l)
        y = 0.5 * t0 * (a + 0.25 * t0^2) + t0
        if abs(y - bhat) < tol
            return (-0.25 * t0^2, s * t0)
        else
            y > bhat ? u = t0 : l = t0
        end
    end

    error("Failed to converge in $(maxiters) steps!")
end

"""
    projection_by_newton(x, y)

Projects the point (x,y) onto the set B = {(p,q) : p + 0.25*q^2 ≤ 0}.

For points already inside B, returns the original point unchanged.
For points outside, projects onto the boundary p + 0.25*q^2 = 0 by solving the
stationarity condition, which reduces to the depressed cubic

    q³ + 4(2+x)q - 8y = 0.

When the discriminant D = (4(2+x)/3)³ + (4y)² ≥ 0 there is one real root,
solved directly via Cardano's formula. When D < 0 there are three real roots
(trigonometric method); the one minimising the squared distance to (x,y) is chosen.

Returns:
- (p, q): projected point satisfying p = -0.25*q²
"""
function projection_by_newton(x, y)
    if x + 0.25 * y^2 ≤ 0
        return (x, y)
    end

    # Depressed cubic: q³ + a*q + b = 0
    a = 4 * (2 + x)
    b = -8 * y
    D = (a / 3)^3 + (b / 2)^2

    if D >= 0
        sq = sqrt(D)
        q  = cbrt(-b/2 + sq) + cbrt(-b/2 - sq)
    else
        # Three real roots via the trigonometric method; pick the closest projection
        m     = 2 * sqrt(-a / 3)
        φ     = acos(clamp(3*b / (a * m), -1.0, 1.0)) / 3
        roots = (m * cos(φ), m * cos(φ - 2π/3), m * cos(φ + 2π/3))
        q     = roots[argmin((-0.25*r^2 - x)^2 + (r - y)^2 for r in roots)]
    end

    return (-0.25 * q^2, q)
end

