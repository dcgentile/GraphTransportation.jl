include("../utils.jl")
using SparseArrays
using LinearAlgebra
using BlockBandedMatrices
"""
this file contains functionality for projecting pairs of discretized curves and vector fields
on graphs onto the set of such pairs which satisfies the (graph, discretized) continuity equation
"""


function form_ceh_system(Q, N)
    """
    let L be the Laplacian associated to Q, and N = 1/h be the number of steps
    given a ρ ∈ V_{n,h}^1, m ∈ V_{e,h}^0, we want to project onto the set of
    (ρ, m) satisfying the Galerkin-discretized discrete continuity equation (see eqn (10)
    in Erbar et al. 2020). A Lagrange mulitpliers argument shows that this can be done by solving
    a linear system Ax=b, where A is, essentially, the "differential operator"
    D = (∂t^2) + L
    Now because the the value of Lf(x) will in general depend on non-local information
    (i.e values of f away from x), care must be taken when setting up this projection problem,
    and we must treat it as one simultaneous linear system on R^{NV + 1 × NV + 1}
    """
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
    o = fill(1. * N^2 * I(V), N - 1)
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
    #return sparse(A)
    #return A
end

function form_b(ρ_A, ρ_B, ρ, m, Q)
    """
    succintly, the target b in the system Ax=b that we need to solve can be written
    b = - ∂tρ - div m

    arguments
    ρ_A, ρ_B ∈ R^n, the initial and terminal measures
    Q, the Markov kernel encoding the graph
    ρ ∈ V_{n,h}^1 (i.e. a matrix of size (1 + 1/h) × n)
    m ∈ V_{e,h}^1 (i.e. a matrix of size (1/h) × n × n)
    """
    N = size(ρ, 1) - 1
    h_inv = N
    divm = similar(ρ[1:N,:])
    @inbounds for t in 1:N
        divm[t,:] = graph_divergence(Q, m[t,:,:])
    end
    ∂tρ = ρ[2:N+1,:] - ρ[1:N,:]
    ∂tρ[1,:] = ρ[2,:] - ρ_A
    ∂tρ[N,:] = ρ_B - ρ[N,:]
    ∂tρ = h_inv * ∂tρ
    v = vec(permutedims(∂tρ .+ divm))
    push!(v, 0) # we need to tack on a zero for technical reasons having to do with solvability of the system. cf the last paragraph of Erbar et al 2020 Sec 4.2
    return -1*v
end


function proj_CE!(ρ, m, μ, ν, Q, A=nothing)
    """
    solves the projection problem in place, updating ρ and m to satisying the Galerkin-discretized discrete continuity equation
    if φ solves the linear system, the update is given by
    ρ[1,x] = ρ_A
    ρ[N+1,x] = ρ_B
    ρ[2:N,x] .+= h^-1*(φ[2:N,:] - φ[1:N-1,:])
    m[1:N,:,:] .+= ∇_G(φ[1:N,:])
    """
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
        m[i,:,:] .+= graph_gradient(Q,φ[i,:])
    end
end

function in_CEplus(ρ, m, μ, ν, Q; verbose=false)
    discrep = maximum(abs.(CE_operator(ρ,m,Q)))
    if verbose
        println("max{∂tρ + ∇⋅m} = $(discrep),  min{ρ} = $(minimum(ρ))")
    end
    return ρ[1,:] == μ && ρ[end,:] == ν &&  discrep < 1e-9 && !any(ρ .< -1e-10)
end



function proj_CENN(ρ, m, μ, ν, Q, A=nothing; verbose=false)
    """
    because our curves are not continuous, it's possible for a sequence of measures representing a rectification of a curve to actually leave the simplex
    while satisfying the continuity equation, so here we employ the POCS routine to identify a curve of strictly non-negative measures
    """
    xρ = ρ
    xm = m

    yρ = zeros(size(ρ))
    ym = zeros(size(m))

    pρ = zeros(size(ρ))
    pm = zeros(size(m))

    qρ = zeros(size(ρ))
    qm = zeros(size(m))
    while !in_CEplus(xρ, xm, μ, ν, Q, verbose=verbose)
        yρ = max.(xρ + pρ, 0)
        ym = xm + pm

        pρ = xρ + pρ - yρ
        pm = xm + pm - ym

        xρ, xm = proj_CE(yρ + qρ, ym + qm, μ, ν, Q, A)

        qρ = yρ + qρ - xρ
        qm = ym + qm - xm
    end
    return (xρ, xm)
end

function proj_CE(ρ, m, μ, ν, Q, A=nothing)
    """
    solves the projection problem returning ρ and m satisying the Galerkin-discretized discrete continuity equation
    if φ solves the linear system, the update is given by
    ρ[1,x] = ρ_A
    ρ[N+1,x] = ρ_B
    ρ[2:N,x] = ρ[2:N,x] +  h^-1*(φ[2:N,:] - φ[1:N-1,:])
    m[1:N,:,:] = m[1:N,:,:] + ∇_G(φ[1:N,:])
    """
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

function find_lagrange_multiplier(ρ, m, μ, ν, Q, A=nothing)
    """
    solves the projection problem returning ρ and m satisying the Galerkin-discretized discrete continuity equation
    if φ solves the linear system, the update is given by
    ρ[1,x] = ρ_A
    ρ[N+1,x] = ρ_B
    ρ[2:N,x] = ρ[2:N,x] +  h^-1*(φ[2:N,:] - φ[1:N-1,:])
    m[1:N,:,:] = m[1:N,:,:] + ∇_G(φ[1:N,:])
    """
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
