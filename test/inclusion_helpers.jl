# Helper functions for testing constraint-set membership.
# Migrated from src/tests/Inclusion.jl.

function is_in_ScriptK(ρ_min, ρ_pl, θ, verbose=false)
    @assert size(ρ_min) == size(ρ_pl) && size(ρ_pl) == size(θ)
    for idx in eachindex(θ)
        x = geomean(ρ_min[idx], ρ_pl[idx])
        y = θ[idx]
        if !(0 ≤ y)
            return false
        end
        if !(y ≤ x || abs(x - y) < 1e-14)
            return false
        end
    end
    verbose && println("Passed Script{K} Inclusion Test")
    return true
end

function is_in_JPM(q, ρ_minus, ρ_plus, verbose=false)
    for (idx, _) in pairs(ρ_minus)
        t, x, y = Tuple(idx)
        ρ_minus[idx] == q[t, x] || return false
        ρ_plus[idx]  == q[t, y] || return false
    end
    verbose && println("Passed inclusion in J_PM")
    return true
end

function is_in_JEq(ρ, q, verbose=false)
    result = isapprox(ρ, q)
    verbose && result && println("Passed Inclusion in J_EQ")
    return result
end

function is_in_JAvg(ρ, ρ_bar, verbose=false)
    A = avg_operator(size(ρ, 1))
    ρ_avg = A * ρ
    result = isapprox(ρ_bar, ρ_avg)
    verbose && result && println("Passed Inclusion in J_Avg")
    return result
end

function is_in_CE_weakly(ρ, m, Q, u, verbose=false)
    N, V = size(ρ)
    ∂tρ = (N - 1) * (ρ[2:N, :] - ρ[1:N-1, :])
    divm = similar(∂tρ)
    @inbounds for i = 1:N-1
        divm[i, :] = graph_divergence(Q, m[i, :, :])
    end
    a = ∂tρ + divm
    φ = rand(N - 1, V)
    ce_discrepancy = sum(a .* φ * u)
    verbose && println("CE discrepancy: $ce_discrepancy")
    return abs(ce_discrepancy) ≤ 1e-10
end
