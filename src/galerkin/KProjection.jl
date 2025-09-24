tolerance = 1e-6

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
    return (ρ_m, ρ_p, θ)
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

"""
    project_by_bisection(a,b,c; tol=tolerance, maxiters=500)

Description of the function.

#TODO
"""
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
    troubleshooter(a,b,c,maxiters)
    error("Failed to find root!")
end


"""
    troubleshooter(a,b,c,maxiters,tol=tolerance)

Description of the function.

#TODO
"""
function troubleshooter(a,b,c,maxiters,tol=tolerance)
    l, u = find_bracket(a,b,c)
    for i in 1:maxiters
        x0 = (l + u) / 2
        f = (-0.5 * c * x0) + ((0.5 * a - b) * √x0) + ((a - 0.5 * b) / √x0) + (0.5 * c / x0)
        if abs(f) < tol
            α = ((a * √x0) + (b / √x0) + c) / (x0 + 1/x0 + 1)
            return α * [√x0; 1/√x0; 1]
        else
            f > 0 ? l=x0 : u=x0
        end
        if i == maxiters
            println(f)
        end
    end
    println([a, b, c])
    l, u = find_bracket(a,b,c)
    println([l,u])
    println(
         (-0.5 * c * l) + ((0.5 * a - b) * √l) + ((a - 0.5 * b) / √l) + (0.5 * c / l)
    )
    println(
         (-0.5 * c * u) + ((0.5 * a - b) * √u) + ((a - 0.5 * b) / √u) + (0.5 * c / u)
    )
    println(x0)
end

"""
    find_bracket(a, b, c, maxiters = 64)

Description of the function.

#TODO
"""
function find_bracket(a, b, c, maxiters = 64)
    s, t = 0, 0
    for n=1:maxiters
        x0 = 1/2^n
        #x1 = 2^n
        f0 = (-0.5 * c * x0) + ((0.5 * a - b) * √x0) + ((a - 0.5 * b) / √x0) + (0.5 * c / x0)
        #println(f0)
        if f0 > 0
            s = x0
            break
        end
        #f1 = (-0.5 * c * x1) + ((0.5 * a - b) * √x1) + ((a - 0.5 * b) / √x1) + (0.5 * c / x1)
        #if !signequal(f0, f1)
            #return (x0, x1)
        #end
    end
    for n = 1:maxiters
        x0 = s * 2^n
        f0 = (-0.5 * c * x0) + ((0.5 * a - b) * √x0) + ((a - 0.5 * b) / √x0) + (0.5 * c / x0)
        if f0 < 0
            t = x0
            return (s,t)
        end
    end
    println([a, b, c])
    error("Could not find bracket")

end

function project_by_newton(x,y,z)
    w(q) = [sqrt(q); 1/sqrt(q); 1]
    q_star = rootfinder(x,y,z)
    w_star = w(q_star)
    τ = ([x;y;z] ⋅ w_star) / (w_star ⋅ w_star)
    return τ * w_star
end

function rootfinder(x, y, z; tol=tolerance, maxiters=100)
    w(q) = [sqrt(q); 1/sqrt(q); 1]
    n(q) = [-1/(2*sqrt(q)); -sqrt(q)/2; 1]
    p = [x; y; z]
    
    # f(q) = p ⋅ (w(q) × n(q)) - we want to find the root of this function
    function f(q)
        wq = w(q)
        nq = n(q)
        cross_product = [
            wq[2]*nq[3] - wq[3]*nq[2],
            wq[3]*nq[1] - wq[1]*nq[3], 
            wq[1]*nq[2] - wq[2]*nq[1]
        ]
        return p ⋅ cross_product
    end
    
    # Compute derivative f'(q) using finite differences
    function f_prime(q)
        h = max(1e-8, 1e-6 * abs(q))  # adaptive step size
        return (f(q + h) - f(q - h)) / (2 * h)
    end
    
    # Generate more comprehensive initial guesses
    base_guesses = [1.0, 0.1, 10.0, 0.01, 100.0, 0.001, 1000.0]
    ratio_guess = x > 0 && y > 0 ? x/y : 1.0
    geometric_guesses = [sqrt(ratio_guess), 1/sqrt(ratio_guess)]
    logarithmic_guesses = [exp(-2), exp(-1), exp(1), exp(2)]
    random_guesses = [0.1 * rand() + 0.01 for _ in 1:5]  # small random perturbations
    
    all_guesses = vcat(base_guesses, [ratio_guess], geometric_guesses, logarithmic_guesses, random_guesses)
    
    # Try each initial guess with adaptive Newton's method
    for (attempt, q_init) in enumerate(all_guesses)
        if q_init <= 0
            continue  # skip invalid initial guesses
        end
        
        q = q_init
        prev_fq = Inf
        stagnation_count = 0
        
        try
            for iter in 1:maxiters
                fq = f(q)
                if abs(fq) < tol
                    return q  # found the root q_star
                end
                
                # Check for stagnation
                if abs(fq - prev_fq) < 1e-12
                    stagnation_count += 1
                    if stagnation_count > 5
                        break  # try next initial guess
                    end
                else
                    stagnation_count = 0
                end
                prev_fq = fq
                
                fpq = f_prime(q)
                if abs(fpq) < 1e-14
                    # Try gradient descent step when derivative is too small
                    q_new = q - 0.01 * sign(fq) * abs(q)
                else
                    # Standard Newton step with damping
                    step = fq / fpq
                    damping = min(1.0, abs(q) / max(abs(step), 1e-10))  # adaptive damping
                    q_new = q - damping * step
                end
                
                # Ensure q stays positive with better bounds
                if q_new <= 0
                    q_new = q * 0.5
                elseif q_new > 1e6  # prevent overflow
                    q_new = 1e6
                end
                
                # Check for oscillation and add perturbation
                if iter > 10 && abs(q_new - q) < 1e-12
                    q_new += 0.01 * rand() * q  # small random perturbation
                end
                
                q = q_new
            end
        catch e
            # If any error occurs during this attempt, try next initial guess
            continue
        end
    end
    
    # Final fallback: try bisection method on a reasonable interval
    try
        return bisection_fallback(f, 1e-6, 1e6, tol, maxiters)
    catch
        error("All rootfinding methods failed to converge for inputs ($x, $y, $z)")
    end
end

# Fallback bisection method for when Newton's method fails completely
function bisection_fallback(f, a, b, tol, maxiters)
    fa, fb = f(a), f(b)
    
    # Ensure we have a sign change
    if fa * fb > 0
        # Try to find a sign change by expanding the interval
        for i in 1:10
            a_new, b_new = a / 10^i, b * 10^i
            try
                fa_new, fb_new = f(a_new), f(b_new)
                if fa_new * fb_new < 0
                    a, b, fa, fb = a_new, b_new, fa_new, fb_new
                    break
                end
            catch
                continue
            end
        end
        
        if fa * fb > 0
            error("Cannot find sign change for bisection")
        end
    end
    
    for _ in 1:maxiters
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tol || abs(b - a) < tol
            return c
        end
        
        if fa * fc < 0
            b, fb = c, fc
        else
            a, fa = c, fc
        end
    end
    
    error("Bisection method failed to converge")
end
