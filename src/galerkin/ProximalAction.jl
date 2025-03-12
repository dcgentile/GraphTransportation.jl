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
    for i in eachindex(θ, m)
        θ[i], m[i] = proj_B(θ[i], m[i])
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
        θ_pr[idx], m_pr[idx] = proj_B(θ[idx], m[idx])
    end

    return (θ_pr, m_pr)
end

function newton_projection(x0, x, y;tol=1e-10, maxiter=500)
    """
    a naive implementation of Newton's method for projecting onto the parabolic set B
    writing out the minimization problem can be expressed as a 1D convex optimization function,
    and it's derivative is given by f --- hence it suffices to find the (unique) zero of f
    """
    p_curr = x0
    i = 1
    for i in 1:maxiter
	    f = 0.125 * p_curr^3 + (1 + 0.5*x)*p_curr - y
        fprime = 0.375 * p_curr^2 + (1 + 0.5*x)
        if abs(fprime) < tol
            return (-(p_curr)^2 / 4, p_curr) # if the derivative is zero, the next iteration will be undefined, so tap out and return
        end
        p_next = p_curr - f / fprime
        d = abs(p_curr - p_next)
        if d < tol
            return (-(p_curr)^2 / 4, p_curr)
        end
        p_curr = p_next
    end
    error("Failed to converge")
end


function proj_B(x, y; maxiter=50, tol=1e-10)
    """
    given real numbers x,y project (x,y) to the set B
    """

    return x + 0.25*y^2 ≤ 0 ? (x, y) : newton_projection(-y^2/8, x, y, maxiter=maxiter, tol=tol)

end
	#if x + 0.25 * y^2 ≤ 0
        ## if you're already in the set, nothing to do
        #return (x, y)
    #end
    ## otherwise, make an initial guess by projecting onto a linear approximation
    ## of the objective function
    ##try
    ##catch error
        ##project_by_GD(x,y)
    ##end
#
#
    ##return newton_projection(x0, x, y)
#
    #x0 = -y^2/8
    #return newton_projection(x0, x, y, maxiter=maxiter, tol=tol)
    ##if x ≤ y
        ##Ex0 = (y - x) / 2
    ##elseif x ≤ -y
        ##Ex0 = (x - y) / 2
    ##end
    ##for initial_guess in collect(-100:100)#[n*x0 for n=1:100]
        ##try
            ##return newton_projection(initial_guess, x, y, maxiter=maxiter, tol=tol)
        ##catch error
            ##continue
        ##end
##
##
    ##end
#end




# Below: various failures of trying to come up with clever, parallelizable, foolproof projection schemes.



#function proj_B_external(x,y)
    #if y == 0
        #return (0,0)
    #elseif y ≥ 0
        #q = find_zero(q -> f(x,y,q), y^2)#(0, y^2), Bisection())
        #return (-0.25 * q^2, q)
    #else
        #q = find_zero(q -> f(x,y,q), -y^2)#(-y^2, 0), Bisection())
        #return (-0.25 * q^2, q)
    #end
#
#end
#
#
#function proj_B_verbose(x, y; maxiter=50, tol=1e-10)
	#if x + 0.25 * y^2 ≤ 0
        #return (x, y)
    #end
    #inits = collect(0.5:0.5:200)
    #for x0 in inits
        #d = Inf
        #p_curr = x0
        #i = 1
        #while d > tol
            #if i > maxiter
#
            #end
	        #f = 0.125 * p_curr^3 + (1 + 0.5*x)*p_curr - y
            #fprime = 0.375 * p_curr^2 + (1 + 0.5*x)
            #if abs(fprime) < tol
                #return (-(p_curr)^2 / 4, p_curr) # if the derivative is zero, the next iteration will be undefined, so tap out and return
            #end
            #p_next = p_curr - f / fprime
            #d = abs(p_curr - p_next)
            #p_curr = p_next
            #i += 1
        #end
        #return (-(p_curr)^2 / 4, p_curr)
    #end
    #error("Failed to converge in proximal action scheme after several initial guesses")
#end
#
#function proj_B_verbose_GPU(x, y; maxiter=50, tol=1e-10)
    ## Handle the trivial case
    #if x + 0.25 * y^2 ≤ 0
        #return (x, y)
    #end
#
    ## Instead of trying multiple initial values, just use one reasonable guess
    #p_curr = abs(y) + 1.0  # A simple starting point
#
    ## Newton's method iteration
    #for _ in 1:maxiter
        #f = 0.125 * p_curr^3 + (1 + 0.5*x)*p_curr - y
        #fprime = 0.375 * p_curr^2 + (1 + 0.5*x)
#
        ## Handle near-zero derivative
        #if abs(fprime) < tol
            #return (-(p_curr)^2 / 4, p_curr)
        #end
#
        #p_next = p_curr - f / fprime
        #d = abs(p_curr - p_next)
#
        ## Check for convergence
        #if d <= tol
            #return (-(p_curr)^2 / 4, p_curr)
        #end
#
        #p_curr = p_next
    #end
#
    ## If we reach here, we didn't converge, but return best guess anyway
    ## instead of throwing an error
    #return (-(p_curr)^2 / 4, p_curr)
#end
#
#function proj_B_Halley(x, y; maxiter=50, tol=1e-10)
    #if x + 0.25 * y^2 ≤ 0
        #return (x, y)
    #end
#
    ## Instead of trying multiple initial values, just use one reasonable guess
    #p_curr = abs(y) + 1.0  # A simple starting point
#
    ## Newton's method iteration
    #for _ in 1:maxiter
        #f = 0.125 * p_curr^3 + (1 + 0.5*x)*p_curr - y
        #f_one = 0.375 * p_curr^2 + (1 + 0.5*x)
        #f_two = 0.75 * p_curr
#
#
        #numer = 2 * f * f_one
        #denom = 2 * f_one^2 - f * f_two
#
        ## Handle near-zero derivative
        #if abs(denom) < tol
            #return (-(p_curr)^2 / 4, p_curr)
        #end
#
        #p_next = numer / denom
        #d = abs(p_curr - p_next)
#
        ## Check for convergence
        #if d <= tol
            #return (-(p_curr)^2 / 4, p_curr)
        #end
#
        #p_curr = p_next
    #end
#
    ## If we reach here, we didn't converge, but return best guess anyway
    ## instead of throwing an error
    #return (-(p_curr)^2 / 4, p_curr)
#end
#
#
