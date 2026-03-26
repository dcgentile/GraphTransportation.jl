"""
    proj_Jpm(q, ρ_minus, ρ_plus, Q) -> (q_proj, ρ_minus_proj, ρ_plus_proj)

Project `(q, ρ_minus, ρ_plus)` onto the set
`J_pm = {(q, ρ_minus, ρ_plus) : ρ_minus[t,x,y] = q[t,x], ρ_plus[t,x,y] = q[t,y]}`.

The projection is computed node-by-node via a closed-form expression derived
from a Lagrange multiplier argument (Erbar et al., section 4.5).
"""
function proj_Jpm(q, ρ_minus, ρ_plus, Q)
    #N, V = size(q)
    #Qprime = reshape(Q, 1, size(Q) ...)
    #α = (1 .+ sum(Q, dims=2)).^-1
    #β = 0.5 * dropdims(sum((ρ_minus .+ permutedims(ρ_plus, (1,3,2))) .* Qprime, dims=3), dims=3)
    #q_proj = (q .+ β) .* α'
    #ρ_minus_proj = reshape(q_proj, N, V, 1)
    #ρ_plus_proj = reshape(q_proj, N, 1, V)

    N, V = size(q)
    q_proj = zeros((N, V))

    for i=1:N, x=1:V
        a = 1 / (1 + sum(Q[x,:]))
        b = q[i,x]
        c = 0.5*(sum((ρ_minus[i,x,:] + ρ_plus[i, :, x]) .* Q[x,:]))
        q_proj[i,x] = a * (b + c)
    end

    ρ_minus_proj = zeros((N, V, V))
    ρ_plus_proj = zeros((N, V, V))

    for i=1:N, x=1:V, y=1:V
        ρ_minus_proj[i, x, y] = q_proj[i, x]
        ρ_plus_proj[i, x, y] = q_proj[i, y]
    end

    return (q_proj, ρ_minus_proj, ρ_plus_proj)

end



"""
    proximal_IJpm_star!(q, ρ_minus, ρ_plus, Q; safe=false) -> (q, ρ_minus, ρ_plus)

Compute in-place the proximal mapping of `IJ_pm*` via Moreau's identity
(Erbar et al., section 4.5):

    prox_{IJ_pm*}(q, ρ_minus, ρ_plus) = (q, ρ_minus, ρ_plus) - proj_{J_pm}(q, ρ_minus, ρ_plus)

The projection `q_proj` is computed node-by-node; the subtraction for
`ρ_minus` and `ρ_plus` broadcasts via `reshape` without materialising
the full `(N, V, V)` projected arrays.
"""
function proximal_IJpm_star!(q, ρ_minus, ρ_plus, Q; safe=false)
    N, V = size(q)
    q_proj = zeros(N, V)

    for i in 1:N, x in 1:V
        a = 1 / (1 + sum(Q[x,:]))
        b = q[i,x]
        c = 0.5 * sum((ρ_minus[i,x,:] + ρ_plus[i,:,x]) .* Q[x,:])
        q_proj[i,x] = a * (b + c)
    end

    if safe
        q_proj_full, ρ_minus_proj, ρ_plus_proj = proj_Jpm(q, ρ_minus, ρ_plus, Q)
        @assert is_in_JPM(q_proj_full, ρ_minus_proj, ρ_plus_proj)
    end

    # ρ_minus_proj[i,x,y] == q_proj[i,x] and ρ_plus_proj[i,x,y] == q_proj[i,y],
    # so the subtractions broadcast directly without materializing the (N,V,V) arrays.
    ρ_minus .-= reshape(q_proj, N, V, 1)
    ρ_plus  .-= reshape(q_proj, N, 1, V)
    q       .-= q_proj

    return (q, ρ_minus, ρ_plus)
end


"""
    proximal_IJpm_star(q, ρ_minus, ρ_plus, Q) -> (q_new, ρ_minus_new, ρ_plus_new)

Non-mutating variant of `proximal_IJpm_star!`: allocates and returns new arrays.
"""
function proximal_IJpm_star(q, ρ_minus, ρ_plus, Q)
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
