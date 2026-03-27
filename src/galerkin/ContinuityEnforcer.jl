"""
    form_ceh_system(Q, N) -> LU

Assemble and factorise the `(NV+1) × (NV+1)` linear system whose solution
gives the Lagrange multiplier for the projection onto the Galerkin-discretised
continuity equation constraint set (Erbar et al. 2020, eqn. 10).

The system matrix is essentially the differential operator `D = ∂t² + L` where
`L` is the graph Laplacian of `Q`.  A mean-zero normalisation constraint is
appended (ones border, zero corner) to ensure a unique solution.

Returns an `LU` factorisation for efficient repeated solves.
"""
function form_ceh_system(Q, N)
	V = size(Q,1)
    #form the Laplacian, and write N copies of it to an array
    L = 1. * laplacian_from_transition(Q)
    Ld = fill(L, N)

    # form the ∂t^2 operator: h^-2 * (f(t + 1) - 2f(t) + f(t - 1))
    # form a diagonal -2*N^2*I, representing the middle summand
    Ad = fill(-2 * N^2 * I(V), N)
    # adjust the terminal matrices to account for a "one-sided" limit
    Ad[1] = -N^2 * I(V)
    Ad[end] = -N^2 * I(V)
    # the diagonal of our final matrix
    d = Ad .+ Ld
    # the off diagonals of our final matrix
    o = fill(Matrix(1. * N^2 * I(V)), N - 1)
    #glue it all together
    A = BlockTridiagonal(o, d, o)
    # in order to ensure a unique solution, an additional lagrange multiplier is added
    # which forces the solution to be a "mean zero function" in space and time
    # this is accomplished by padding the matrix on the right and the bottom by ones,
    # and setting the very last entry to 0
    A = cat(A, ones(N*V)', dims=1)
    A = cat(A, ones(N*V + 1), dims=2)

    A[end, end] = 0
    return lu(sparse(A))
end

"""
    form_b(ρ_A, ρ_B, ρ, m, Q) -> Vector

Compute the right-hand side `b = -(∂tρ + div m)` for the continuity-equation
projection linear system.

# Arguments
- `ρ_A`, `ρ_B`: initial and terminal node measures (length-`V` vectors)
- `ρ`: density curve, matrix of size `(N+1) × V`
- `m`: edge-flux tensor of size `N × V × V`
- `Q`: Markov transition matrix defining the graph
"""
function form_b(ρ_A, ρ_B, ρ, m, Q)
    N = size(ρ, 1) - 1
    V = size(Q, 1)
    h_inv = N
    divm = zeros(N, V)
    @inbounds for t in 1:N
        graph_divergence!(@view(divm[t,:]), Q, @view(m[t,:,:]))
    end
    ∂tρ = ρ[2:N+1,:] - ρ[1:N,:]
    ∂tρ[1,:] = ρ[2,:] - ρ_A
    ∂tρ[N,:] = ρ_B - ρ[N,:]
    ∂tρ = h_inv * ∂tρ
    v = vec(permutedims(∂tρ .+ divm))
    push!(v, 0) # we need to tack on a zero for technical reasons having to do with solvability of the system. cf the last paragraph of Erbar et al 2020 Sec 4.2
    return -1*v
end


"""
    proj_CE!(ρ, m, μ, ν, Q, A=nothing) -> (ρ, m)

Project `(ρ, m)` in-place onto the Galerkin-discretised continuity equation
constraint set.  If `φ` solves the linear system `A·φ = b`, the update is:

    ρ[1,:]   .= μ
    ρ[2:N,:] .+= N·(φ[2:N,:] - φ[1:N-1,:])
    ρ[N+1,:] .= ν
    m[t,:,:] .+= ∇_G(φ[t,:])   for each t

`A` is the factorised system from `form_ceh_system`; if omitted it is assembled
on the fly (expensive — pass a cached factorisation when possible).
"""
function proj_CE!(ρ, m, μ, ν, Q, A=nothing)
    N, V, _ = size(m)
    if isnothing(A)
        A = form_ceh_system(Q, N)
    end

    b = form_b(μ, ν, ρ, m, Q)
    ϕ = A \ b
    # reshape the solution to be compatible with ρ
    φ = reshape(ϕ[1:end-1], V, N)'
    # update ρ
    ρ[1,:] .= μ
    ρ[2:N,:] .+= N * (φ[2:N,:] - φ[1:N-1,:])
    ρ[N+1,:] .= ν
    # update m
    @inbounds for i in 1:N
        add_graph_gradient!(@view(m[i,:,:]), Q, @view(φ[i,:]))
    end
    return (ρ, m)
end

"""
    in_CEplus(ρ, m, μ, ν, Q; verbose=false) -> Bool

Return `true` if `(ρ, m)` satisfies the continuity equation with boundary
conditions `μ`, `ν` and has non-negative densities (tolerance `1e-10`).
Prints the max discrepancy and minimum density when `verbose=true`.
"""
function in_CEplus(ρ, m, μ, ν, Q; verbose=false)
    discrep = maximum(abs.(CE_operator(ρ,m,Q)))
    if verbose
        println("max{∂tρ + ∇⋅m} = $(discrep),  min{ρ} = $(minimum(ρ))")
    end
    return ρ[1,:] == μ && ρ[end,:] == ν &&  discrep < 1e-9 && !any(ρ .< -1e-10)
end



"""
    proj_CENN(ρ, m, μ, ν, Q, A=nothing; verbose=false, maxiters=2^32) -> (ρ, m)

Project onto the intersection of the continuity-equation constraint set and the
non-negativity constraint via the POCS (Projections onto Convex Sets) algorithm.

Because discretised curves are not continuous, a continuity-equation-satisfying
sequence can still leave the probability simplex; this POCS loop enforces both
constraints simultaneously.
"""
function proj_CENN(ρ, m, μ, ν, Q, A=nothing; verbose=false, maxiters=2^32)
    xρ = ρ
    xm = m

    yρ = zeros(size(ρ))
    ym = zeros(size(m))

    pρ = zeros(size(ρ))
    pm = zeros(size(m))

    qρ = zeros(size(ρ))
    qm = zeros(size(m))

    p = ProgressUnknown(spinner=true)

    for i in 1:maxiters
        if !in_CEplus(xρ, xm, μ, ν, Q, verbose=verbose)
            yρ = max.(xρ + pρ, 0)
            ym = xm + pm
            pρ = xρ + pρ - yρ
            pm = xm + pm - ym

            xρ, xm = proj_CE(yρ + qρ, ym + qm, μ, ν, Q, A)

            qρ = yρ + qρ - xρ
            qm = ym + qm - xm
            discrep = maximum(abs.(CE_operator(xρ,xm,Q)))
            next!(p; showvalues=[("max{∂tρ + ∇⋅m} = ", discrep), ("min{ρ} = ", minimum(xρ)), ("Current iteration", i)])
        else
            break
        end
    end
    return (xρ, xm)
end

"""
    proj_CE(ρ, m, μ, ν, Q, A=nothing) -> (ρ_new, m_new)

Non-mutating variant of `proj_CE!`: allocates and returns new arrays.
"""
function proj_CE(ρ, m, μ, ν, Q, A=nothing)
    N, V, _ = size(m)
    if isnothing(A)
        A = form_ceh_system(convert(Array, Q), N)
    end
    b = form_b(μ, ν, ρ, m, Q)
    ϕ = A \ b
    φ = reshape(ϕ[1:end-1], V, N)'
    ρ_pr = copy(ρ)
    ρ_pr[1,:] .= μ
    ρ_pr[2:N,:] .+= N .* (φ[2:N,:] .- φ[1:N-1,:])
    ρ_pr[N+1,:] .= ν
    m_pr = similar(m)
    @inbounds for i=1:N
        m_pr[i,:,:] = m[i,:,:] + graph_gradient(Q, φ[i,:])
    end

    return (ρ_pr, m_pr)
end

"""
    find_lagrange_multiplier(ρ, m, μ, ν, Q, A=nothing) -> (b, ϕ, φ, ∂tφ, ∇φ)

Solve the continuity-equation projection system and return the intermediate
quantities `(b, ϕ, φ, ∂tφ, ∇φ)` for diagnostic or debugging purposes.
"""
function find_lagrange_multiplier(ρ, m, μ, ν, Q, A=nothing)
    N, V, _ = size(m)
    if isnothing(A)
        A = form_ceh_system(convert(Array, Q), N)
    end
    b = form_b(μ, ν, ρ, m, Q)
    ϕ = A \ b
    φ = reshape(ϕ[1:end-1], V, N)'
    ∂tφ = N * (φ[2:N,:] - φ[1:N-1,:])
    ∇φ = similar(m)
    @inbounds for i=1:N
        ∇φ =  graph_gradient(Q, φ[i,:])
    end

    return (b, ϕ, φ, ∂tφ, ∇φ)
end
