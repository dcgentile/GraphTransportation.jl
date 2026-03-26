"""
    form_avg_system(N) -> LU

Assemble and factorise the `N × N` tridiagonal matrix for the time-averaging
projection linear system (Erbar et al. 2020, section 4.6).

The matrix depends only on the step count `N = 1/h` and is constant across all
nodes, so this should be called once when constructing an `ErbarCache`.
Returns an `LU` factorisation.
"""
function form_avg_system(N)
    off_diag = ones(N - 1)
    diag = 6 * ones(N)
    diag[1] = 5
    diag[N] = 5
    M = Tridiagonal(off_diag, diag, off_diag)
    #return 0.25 * M
    #return sparse(0.25 * M)
    return lu(sparse(0.25 * M))
end

"""
    form_b(ρ, ρ_bar, ρ_A, ρ_B) -> Matrix

Compute the right-hand side matrix `B` for the averaging-projection linear
system (last unnumbered equation in Erbar et al. 2020, section 4.6):

    B[1,:]   = ρ_bar[1,:]   - (1/2)·(ρ_A      + ρ[2,:])
    B[i,:]   = ρ_bar[i,:]   - (1/2)·(ρ[i+1,:] + ρ[i,:])   for 2 ≤ i ≤ N-2
    B[N-1,:] = ρ_bar[N-1,:] - (1/2)·(ρ_B      + ρ[N-1,:])

All `V` node columns are computed simultaneously.
"""
function form_b(ρ, ρ_bar, ρ_A, ρ_B)
    N = size(ρ, 1)
    B = copy(ρ_bar)
    B[1,:] .-= 0.5 * (ρ_A + ρ[2,:])
    B[2:N-2,:] .-= 0.5 * (ρ[3:N-1,:] + ρ[2:N-2,:])
    B[N-1,:] .-= 0.5 * (ρ_B + ρ[N - 1,:])

    return B
end

"""
    proj_Javg(ρ, ρ_bar, ρ_A, ρ_B, M; safe=false) -> (ρ_proj, ρ_bar_proj)

Project `(ρ, ρ_bar)` onto the set `J_avg = {(ρ, ρ_bar) : ρ_bar = avg_h(ρ)}`.
`M` is the factorised system from `form_avg_system`.  If `safe=true`, asserts
that the projection satisfies the averaging constraint.
"""
function proj_Javg(ρ, ρ_bar, ρ_A, ρ_B, M; safe=false)

    N, V = size(ρ_bar)

    B = form_b(ρ, ρ_bar, ρ_A, ρ_B)
    Λ = M \ B

    ρ_proj = similar(ρ)
    ρ_proj[1,:] = ρ_A
    ρ_proj[2:N,:] = ρ[2:N,:] + 0.5 * (Λ[1:N-1,:] + Λ[2:N,:])
    ρ_proj[N + 1,:] = ρ_B

    ρ_bar_proj = ρ_bar - Λ

    if safe
        try
            A = avg_operator(size(ρ, 1))
            @assert isapprox(A*ρ_proj, ρ_bar_proj)
        catch error
            println("Failed projection to J_avg")
        end
    end

    return (ρ_proj, ρ_bar_proj)


end

"""
    prox_IJavg_star(ρ, ρ_bar, ρ_A, ρ_B, M; safe=false) -> (ρ_new, ρ_bar_new)

Compute the proximal mapping of `IJ_avg*` via Moreau's identity:

    prox_{IJ_avg*}(ρ, ρ_bar) = (ρ, ρ_bar) - proj_{J_avg}(ρ, ρ_bar)

# Arguments
- `ρ`: density curve, matrix of size `(N+1) × V`
- `ρ_bar`: time-averaged density, matrix of size `N × V`
- `ρ_A`, `ρ_B`: boundary measures (length-`V` vectors)
- `M`: factorised system from `form_avg_system`
"""
function prox_IJavg_star(ρ, ρ_bar, ρ_A, ρ_B, M; safe=false)
    #N, V = size(ρ)
    #B = form_b(ρ, ρ_bar, ρ_A, ρ_B)
	#Λ = M \ B
#
    #ρ_bar_pr = ρ_bar - Λ
    #Λ_bar = vcat(zeros(V)', 0.5 * (Λ[2:N-1,:] + Λ[1:N-2,:]), zeros(V)')
    #ρ_pr = ρ + Λ_bar
    #ρ_pr[1,:] = ρ_A
    #ρ_pr[N,:] = ρ_B
    #return (ρ - ρ_pr, ρ_bar - ρ_bar_pr)

    ρ_proj, ρ_bar_proj = proj_Javg(ρ, ρ_bar, ρ_A, ρ_B, M)
    if safe
        @assert is_in_JAvg(ρ_proj, ρ_bar_proj)
    end
    return (ρ - ρ_proj, ρ_bar - ρ_bar_proj)

end

"""
    prox_IJavg_star!(ρ, ρ_bar, ρ_A, ρ_B, M) -> (ρ, ρ_bar)

In-place variant of `prox_IJavg_star`.
"""
function prox_IJavg_star!(ρ, ρ_bar, ρ_A, ρ_B, M)
    #N, V = size(ρ)
    #B = form_b(ρ, ρ_bar, ρ_A, ρ_B)
	#Λ = M \ B

    #ρ[1,:] .-= ρ_A
    #ρ[2:N-1,:] .= (-0.5 * (Λ[1:N-2,:] + Λ[2:N-1,:]))
    #ρ[N,:] .-= ρ_B
#
    #ρ_bar .= Λ

    ρ_proj, ρ_bar_proj = proj_Javg(ρ, ρ_bar, ρ_A, ρ_B, M)
    ρ .= ρ - ρ_proj
    ρ_bar .= ρ_bar - ρ_bar_proj
    return (ρ, ρ_bar)
end
