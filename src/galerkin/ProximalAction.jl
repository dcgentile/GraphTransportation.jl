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
    project_by_bisection(a,b; tol=1e-8, maxiters=2^16)

Description of the function.

#TODO
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
    projection_by_newton(x, y; tol=1e-8, maxiters=100)

Projects the point (x,y) onto the parabola {(p,q) : p + 0.25 * q^2 ≤ 0} using Newton's method.

For points already inside the feasible region, returns the original point.
For points outside, projects onto the boundary p + 0.25 * q^2 = 0.

Arguments:
- x, y: coordinates of the point to project
- tol: convergence tolerance (default 1e-8)  
- maxiters: maximum number of Newton iterations (default 100)

Returns:
- (p, q): projected point on the parabola
"""
function projection_by_newton(x, y; tol=1e-5, maxiters=100, safe=false)
    if x + 0.25 * y^2 ≤ 0
        return (x, y)
    end
    
    # Project onto boundary: minimize |||(p,q) - (x,y)|||^2 subject to p + 0.25*q^2 = 0
    # Using Lagrange multipliers: L = (p-x)^2 + (q-y)^2 + λ(p + 0.25*q^2)
    # ∇L = 0 gives: p = x - λ, q = y - 0.5*λ*q, p + 0.25*q^2 = 0
    # Substituting: (x - λ) + 0.25*(y - 0.5*λ*q)^2 = 0
    # This reduces to finding q, then p = -0.25*q^2, λ = x + 0.25*q^2
    
    # Newton's method to solve: g(q) = q - y + 0.5*(x + 0.25*q^2)*q = 0
    q = y  # initial guess
    
    for _ in 1:maxiters
        g = q - y + 0.5 * (x + 0.25 * q^2) * q
        if abs(g) < tol
            p = -0.25 * q^2

            if safe
                try
                    @assert (x - p) * (-q/2) + (y - q) < tol
                catch
                    println((x - p) * (-q/2) + (y - q))
                end
            end
            return (p, q)
        end
        
        # g'(q) = 1 + 0.5*x + 0.375*q^2
        g_prime = 1 + 0.5 * x + 0.375 * q^2
        
        if abs(g_prime) < 1e-14
            error("Newton's method: derivative too small")
        end
        
        q = q - g / g_prime
    end

    return project_by_bisection(x, y)

    #error("Newton's method failed to converge in $(maxiters) iterations")
end
