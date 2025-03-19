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
        θ[i], m[i] = project_by_bisection(θ[i], m[i])
    end
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


function project_by_bisection(a,b; tol=1e-8, maxiters=100)
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
