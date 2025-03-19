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
    #if 0 ≤ z && z < logmean(x,y)
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
        #return project_by_newton(x, y, z)
        #return project_by_GD(x,y,z)
        #return proj_Ktop(x, y, z)
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

function project_by_newton(a,b,c; tol=tolerance, maxiters=50)
    # we would like to choose an initial condition which projects onto a linearization of the constraint set
    println("a = $(a);\nb = $(b);\n c=$(c);\n")
    l, u = find_bracket(a,b,c)
    println([l, u])
    #x0 = (l + u) / 2
    x0 = l
    d = Inf
    for _ in 1:maxiters
        if d < tol
            α = ((a * √x0) + (b / √x0) + c) / (x0 + 1/x0 + 1)
            return α * [√x0; 1/√x0; 1]
        end
        f = (-0.5 * c * x0) + ((0.5 * a - b) * √x0) + ((a - 0.5 * b) / √x0) + (0.5 * c / x0)
        fprime = (-0.5 * c)  + (0.5 * (0.5 * a + b) / √x0) - (0.5 * (a - 0.5 * b) / (√x0^3)) - (0.5 * c / x0^2)
        x1 = x0 - f/fprime
        d = abs(x1 - x0)
        x0 = x1
    end
    error("Newton failed to converge in $(maxiters) iterations")
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

function project_by_GD(x,y,z; h=1e-3, tol=tolerance, maxiter=50000)
    """
    if (x,y,z) is "above" the surface K, then projecting onto it can be solved via gradient descent
    TODO: gradient descent is terribly slow, so it would be good to solve this via Newton's method
    the problem with Newton's method is the choice of a good initial guess...
    """
    x0 = [sqrt(z); sqrt(z);]
    x1 = x0
    for n in 1:maxiter
        s,t = x0[1], x0[2]
        ∇f = [
            -x + s + 0.5 * t - 0.5 * z*sqrt(t/s);
            -y + t + 0.5 * s - 0.5 * z*sqrt(s/t)
        ]
        x1 = x0 .- h*∇f
        if LinearAlgebra.norm(x1 - x0) < tol
            v = [x1[1]; x1[2]; sqrt(x1[1]*x1[2])]
            return v
        end
        x0 = x1
    end
    println("GD failed to converge with final iteration: $(x1)")
    error("Gradient Descent did not converge")

end


## Functionality for solving the projection problem via
## Newton's method, currently not contributing

α(q::Number) = 0.5*sqrt(q)
β(q::Number) = 0.5* (1 / sqrt(q))
w(q::Number) = [sqrt(q); 1/sqrt(q); 1]
n(q::Number) = [-0.5 / sqrt(q), (-0.5) * sqrt(q), 1 ]
f(p,q::Number) = dot(p, cross(w(q), n(q)))

function proj_Ktop_newton(x,y,z)
    p = [x; y; z]
    q_star = 1
    try
        q_star = find_zero(q -> f(p,q), 4296.34)
        #q_star = find_unique_root(x,y,z)
    catch e
        println(p)
        error("Failed to find root")
    end
    w_star = w(q_star)
    τ = dot(p, w_star / dot(w_star, w_star))
    return τ * w_star
end


## Functionality forn



#function solve_q(x, y)
    #x ≥ y ? find_zero(t -> α(t) - x, x) : find_zero(t -> β(t) - y, y)
#end

## Some needed functions, defined here so that we only compile them the once

#α(q::Number; tol=tolerance) = abs(1 - q) < tol ? 0.5 : (q - log(q) - 1) / (log(q)^2)
#β(q::Number; tol=tolerance) = abs(1 - q) < tol ? 0.5 : α(1 / q)
#α_prime(q::Number; tol=1e-5) = abs(q - 1) < tol ? 1/6 : (2 * (1 - q) + log(q) * (1 + q)) / (q * (log(q)^3))
#α_prime(q::Number; tol=tolerance) = abs(q - 1) < tol ? 1/6 : (-2*q + q*log(q) + log(q) + 2) / (q*log(q)^3)
#β_prime(q::Number; tol=tolerance) = abs(q - 1) < tol ? -1/6 : (2 * (q - 1) - log(q) * (1 + q)) / (q^2 * (log(q)^3))
#w(q::Number) = [q^0.5; q^-0.5; logmean(q^0.5, q^-0.5)]
#n(q::Number) = [-logmean_partial_s(q^0.5, q^-0.5); -logmean_partial_t(q^0.5, q^-0.5); 1]
#fprime(p,q::Number,h) = (f(p, q + h) - f(p, q - h)) / (2*h)

# test for inclusion in the super differential at 0 of the logarithmic mean
#function super_differential_inclusion(s, t)
    #if minimum([s t]) ≤ 0 || maximum([[s t]) < 0.5
        #return false
    #end
#
    #q = solve_q(s, t)
    #return s > t ? t ≥ β(q) : t ≥ α(q)
#end

# solve α(q) - x = 0 in a GPU friendly way (i.e. no try catch blocks)
#function α_root(x; maxiter=500, tol=1e-6)
    #x0 = 1.0
    #for _ in 1:maxiter
        #if x0 < 0
            #x0 = rand() + 0.5 # if our iterate ever suggests a step < 0, we've lost the plot, we we start over with a random number ∈ [0.5, 1.5]
        #end
        #f = α(x0) - x
        #fprime = α_prime(x0)  # we could check for zero derivative here, but α is easy enough to invert that we can be a little lazy
        #x1 = x0 - f/fprime
        #if abs(x1 - x0) < tol
            #return x1
        #end
        #x0 = x1
    #end
#end
#
## solve β(q) - x = 0 in a GPU friendly way (i.e., no try catch blocks)
#function β_root(x; maxiter=500, tol=1e-6)
    #x0 = 1.0
    #for _ in 1:maxiter
        #if x0 < 0
            #x0 = rand() + 0.5 # if our iterate ever suggests a step < 0, we've lost the plot, we we start over with a random number ∈ [0.5, 1.5]
        #end
        #f = α(x0) - x
        #fprime = α_prime(x0)  # we could check for zero derivative here, but α is easy enough to invert that we can be a little lazy
        #x1 = x0 - f/fprime
        #if abs(x1 - x0) < tol
            #return x1
        #end
        #x0 = x1
    #end
#end


#function newton(f::Function, x0::Number, fprime::Function, args::Tuple=();
                #tol::AbstractFloat=1e-8, maxiter::Integer=50, eps::AbstractFloat=1e-10)
    #for _ in 1:maxiter
        #yprime = fprime(x0, args...)
        #if abs(yprime) < eps
            #return x0
        #end
        #y = f(x0, args...)
        #x1 = x0 - y/yprime
        #if abs(x1-x0) < tol
            #return x1
        #end
        #x0 = x1
    #end
    #error("Max iteration exceeded")
#end
#
