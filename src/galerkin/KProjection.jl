using Base: signequal
using LinearAlgebra
include("../utils.jl")

tolerance = 1e-10

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
    p = projection((ρ_minus[I], ρ_plus[I], θ[I]))
    I.e, for each index,
    ρ_m[idx], ρ_p[idx], θ[idx] = proj_K(ρ_m[idx], ρ_p[idx], θ[idx])
    """
    @inbounds for idx in eachindex(θ)
        ρ_m[idx], ρ_p[idx], θ[idx] = proj_K(ρ_m[idx], ρ_p[idx], θ[idx])
    end
end

function proj_K(x, y, z)
    """
    for real numbers x,y,z, we project to the convex set K
    """
    if 0 ≤ z && z < geomean(x,y)
        """
        if this is true, the point is already in K and we are done
        """
        return (x, y, z)
    elseif z ≤ 0
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
        return project_by_bisection(x,y,z)
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

function project_by_bisection(a,b,c; tol=tolerance, maxiters=500)
    l, u = find_bracket(a,b,c)
    x0 = 0
    for _ in 1:maxiters
        x0 = (l + u) / 2
        f = (-0.5 * c * x0) + ((0.5 * a - b) * √x0) + ((a - 0.5 * b) / √x0) + (0.5 * c / x0)
        if abs(f) < tol
            α = ((a * √x0) + (b / √x0) + c) / (x0 + 1/x0 + 1)
            return α * [√x0; 1/√x0; 1]
        else
            f > 0 ? l=x0 : u=x0
        end
    end
    println([a, b, c])
    l, u = find_bracket(a,b,c)
    println([l,u])
    println(x0)
    error("Failed to find root!")
end

function find_bracket(a, b, c, maxiters = 32)
    for n=1:maxiters
        x0 = 1/2^n
        x1 = 2^n
        f0 = (-0.5 * c * x0) + ((0.5 * a - b) * √x0) + ((a - 0.5 * b) / √x0) + (0.5 * c / x0)
        f1 = (-0.5 * c * x1) + ((0.5 * a - b) * √x1) + ((a - 0.5 * b) / √x1) + (0.5 * c / x1)
        if !signequal(f0, f1)
            return (x0, x1)
        end
    end
    error("Could not find bracket")

end
