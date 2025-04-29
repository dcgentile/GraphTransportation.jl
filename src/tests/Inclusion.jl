include("../utils.jl")

function is_in_ScriptK(ρ_min, ρ_pl, θ)
    @assert size(ρ_min) == size(ρ_pl) && size(ρ_pl) == size(θ)
    c = 0
    for idx in eachindex(θ)
        x = geomean(ρ_min[idx], ρ_pl[idx])
        y = θ[idx]
        try
            @assert 0 ≤ y && y ≤ x
        catch AssertionError
            println([y, x, x - y])
            c += 1
        end
    end
    println(c)
end

function is_in_JPM(q, ρ_minus, ρ_plus)
    """
    given q ∈ V_{n,h}^{0} and ρ_min, ρ_plus ∈ V_{e, h}^{0}
    checy that the ρ_min[t,x,y] == q[t,x] and ρ_plus[t,x,y] == q[t,y]
    """
    for (idx, _) in pairs(ρ_minus)
        t, x, y = Tuple(idx);
        try
            @assert ρ_minus[idx] == q[t,x]
        catch e
            println("Failed inclusion in J_PM with error $(e)")
        end
        try
            @assert ρ_plus[idx] == q[t,y]
        catch e
            println("Failed inclusion in J_PM with error $(e)")
        end
    end
end

function is_in_JEq(ρ, q)
    try
        @assert isapprox(ρ, q)
    catch e
        println("Failed inclusion in J_EQ with error $(e)")
    end
end

function is_in_JAvg(ρ, ρ_bar)
    A = avg_operator(size(ρ,1))
    println(size(ρ))
    println(size(A))
    println(size(ρ_bar))
    ρ_avg = A * ρ
    try
	    @assert isapprox(ρ_bar, ρ_avg)
    catch
        for idx in eachindex(ρ_bar)
            if abs(ρ_bar[idx] - ρ_avg[idx]) > 1e-20
                println([abs(ρ_bar[idx] - ρ_avg[idx]), ρ_bar[idx], ρ_avg[idx]])
            end
        end
    end

end

function is_in_CE_weakly(ρ, m, Q, u)
    N, V = size(ρ)
    ∂tρ = (N - 1) * (ρ[2:N,:] - ρ[1:N - 1,:])
    divm = similar(∂tρ)
    @inbounds for i=1:N-1
        divm[i,:] = graph_divergence(Q, m[i,:,:])
    end
    a = ∂tρ + divm
    #φ = ones(N-1, V)
    φ = rand(N-1, V)
    ce_discrepancy = sum(a .* φ * u)
    try
	    @assert ce_discrepancy ≈ 0
    catch err
        println(err)
        println(ce_discrepancy)
    end

end

function CE_operator(ρ, m, Q)
    N, V = size(ρ)
    ∂tρ = (N - 1) * (ρ[2:N,:] - ρ[1:N - 1,:])
    divm = similar(∂tρ)
    @inbounds for i=1:N-1
        divm[i,:] = graph_divergence(Q, m[i,:,:])
    end
    a = ∂tρ + divm
    return a
end

function ∂t(ρ)
    N, V = size(ρ)
    return (N - 1) * (ρ[2:N,:] - ρ[1:N-1,:])
end
