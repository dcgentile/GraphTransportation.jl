
"""
    barycenter(M, weights, Q, h=0.1, maxiters=100, tol=1e-8)

Description of the function.

#TODO
"""
function barycenter(M, weights, Q, h=0.1, maxiters=100, tol=1e-8)
    ν_old = ones(size(G,1)) # inital condition of flow
    ν_new = ones(size(G,1)) # inital condition of flow
    for i=1:maxiters
        δJ = step_direction(ν, M, weights)
        ν_new .= ν_old .+ h * graph_divergence(Q, δJ)

        if norm(ν_new - ν_old) < tol
            break
        end
    end
    return ν_new
end

"""
    step_direction(ν, M, weights, Q)

Description of the function.

#TODO
"""
function step_direction(ν, M, weights, Q)
    tangent_vector = zeros(size(Q))
    p = size(M, 2)
    for i=1:p
        gamma, _ = BBD(Q, ν, M[:, i])
        tangent_vector = tangent_vector + weights[i] * (gamma.vector.m[0,:,:])
    end
    return tangent_vector
end


"""
    coordinates(ν, M, Q, maxiters=100, tol=1e-8, h=0.01)

Description of the function.

#TODO
"""
function coordinates(ν, M, Q, maxiters=100, tol=1e-8, h=0.01)
    p = size(M,2)
    coords_new = (1/p) * ones(p)
    coords_old = (1/p) * ones(p)
    for i=1:maxiters
        coords_new =
        if norm(coords_new - coords_old) < tol
            break
        end

    end
    return coords_new

end
