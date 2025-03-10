function is_in_J_PM(q, ρ_minus, ρ_plus)
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
    return true
end

function is_in_J_eq(ρ, q)
    try
        @assert isapprox(ρ, q)
    catch e
        println("Failed inclusion in J_EQ with error $(e)")
    end
    return true
end

function is_in_J_avg(ρ, ρ_bar)
    ρ_avg = avg_operator(ρ)
    return isapprox(ρ_avg, ρ_bar)

end

function is_in_CE(ρ, m)

end
