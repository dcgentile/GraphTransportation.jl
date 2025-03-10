function proximal_IJpm_star!(q, ρ_minus, ρ_plus, Q)
    """
    compute in place the IJpm, as described in section 4.5 of Erbar et al 2020
    because IJ_pm^star if the fenchel conjugate of an indicator function, it can
    be computed via Moreau's identity: prox_IJpm_star = identity - proj_IJpm
    thus we solve the problem of projecting onto IJpm, and subtract the resultant
    projection off of the argument

    arguments
    ρ ∈ V_{n × h}^0 (i.e. it is a matrix of size 1/h x n, where 1/h is the step size and n the number of nodes)
    ρ_minus ∈ V_{e × h}^0 (i.e. it is a tensor of size 1/h x n × n, where 1/h is the step size and n the number of nodes)
    ρ_plus ∈ V_{e × h}^0 (i.e. it is a tensor of size 1/h x n × n, where 1/h is the step size and n the number of nodes)
    """
    N, V = size(q)
    Qprime = reshape(Q, 1, size(Q) ...)
    α = (1 .+ sum(Q, dims=2)).^-1
    β = 0.5 * dropdims(sum((ρ_minus .+ permutedims(ρ_plus, (1,3,2))) .* Qprime, dims=3), dims=3)
    q_pr = (q .+ β) .* α'

    ρ_minus .-= reshape(q_pr, N, V, 1)
    ρ_plus .-= reshape(q_pr, N, 1, V)
    q .-= q_pr

end


function proximal_IJpm_star(q, ρ_minus, ρ_plus, Q)
    """
    compute the proximal mapping of  IJpm_star, as described in section 4.5 of Erbar et al 2020
    because IJ_pm^star if the fenchel conjugate of an indicator function, it can
    be computed via Moreau's identity: prox_IJpm_star = identity - proj_IJpm
    thus we solve the problem of projecting onto IJpm, and subtract the resultant
    projection off of the argument

    arguments
    ρ ∈ V_{n × h}^0 (i.e. it is a matrix of size 1/h x n, where 1/h is the step size and n the number of nodes)
    ρ_minus ∈ V_{e × h}^0 (i.e. it is a tensor of size 1/h x n × n, where 1/h is the step size and n the number of nodes)
    ρ_plus ∈ V_{e × h}^0 (i.e. it is a tensor of size 1/h x n × n, where 1/h is the step size and n the number of nodes)
    """
    N, V = size(q)
    ρ_pr = similar(q)
    ρ_minus_pr = similar(ρ_minus)
    ρ_plus_pr = similar(ρ_plus)

    #TODO: change these for loops to not work via indexing
    for x in 1:V
        α = 1 / (1 + sum(Q[x,:]))
        for t in 1:N
            a = ρ_minus[t,:,:] .* Q
            b = ρ_plus[t,:,:] .* Q
            β = 0.5 * sum(a[x,:] .+ b[:,x])
            ρ_pr[t,x] = α * (q[t,x] + β)
        end
    end

    for (idx, _) in pairs(ρ_minus)
        t, x, y = Tuple(idx)
        ρ_minus_pr[idx] = ρ_pr[t,x]
        ρ_plus_pr[idx] = ρ_pr[t,y]
    end

    return (q - ρ_pr, ρ_minus - ρ_minus_pr, ρ_plus - ρ_plus_pr)

end
