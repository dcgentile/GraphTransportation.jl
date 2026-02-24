"""
    regularize_cost(matrix, epsilon)

For a given cost matrix and regularization parameter epsilon,
return exp(-C/epsilon), exponentiation performed componentwise.
We perform this regularization so that the entropically regularized
Wasserstein distance can be expressed in terms of a KL divergence.
"""
function regularize_cost(matrix, epsilon)
    return exp.(-matrix ./ epsilon)
end


"""
    logarithmic_change_of_variable(coords)

Project a vector onto the simplex via softmax (logarithmic change of variable).
"""
function logarithmic_change_of_variable(coords)
    v = exp.(coords)
    n = norm(v, 1)
    return v ./ n
end


"""
    sqeuc_loss_grad(p, q)

Gradient of the squared Euclidean loss.
"""
function sqeuc_loss_grad(p, q)
    return p .- q
end


"""
    ell_one_loss(p, q)

Gradient of the L1 loss.
"""
function ell_one_loss(p, q)
    return sign.(p .- q)
end


"""
    kl_loss(p, q)

Gradient of the KL divergence.
"""
function kl_loss(p, q)
    return log.(p ./ q)
end


"""
    sqeuc_loss(p, q)

Squared Euclidean loss between histograms p and q.
"""
function sqeuc_loss(p, q)
    return 0.5 * norm(p .- q, 2)^2
end


"""
    sinkhorn_differentiate(coords, measures, target, cost, epsilon, iters)

Core Sinkhorn fixed-point iteration implementing the Bonneel et al. algorithm.
Returns the Wasserstein barycenter `p` and, if `target` is not `nothing`, the
gradient `w` of the barycentric loss w.r.t. the coordinates.

ARGS
- `coords`: weight vector of length num_measures, should sum to 1
- `measures`: matrix of size (num_nodes, num_measures), each column a probability measure
- `target`: target histogram of length num_nodes, or `nothing` to skip gradient
- `cost`: cost matrix of size (num_nodes, num_nodes)
- `epsilon`: regularization parameter
- `iters`: number of Sinkhorn iterations
"""
function sinkhorn_differentiate(coords, measures, target, cost, epsilon, iters)
    num_nodes, num_measures = size(measures)
    b = ones(num_nodes, num_measures, iters)
    w = zeros(num_measures)
    r = zeros(num_nodes, num_measures)
    phi = Array{Float64}(undef, num_nodes, num_measures, iters)
    k = regularize_cost(cost, epsilon)
    p = zeros(num_nodes)

    for l in 2:iters
        for s in 1:num_measures
            m = measures[:, s]
            b_m = b[:, s, l-1]
            kb_m = k * b_m
            ratio = m ./ kb_m
            phi[:, s, l] = k' * ratio
        end
        p = exp.(log.(phi[:, :, l]) * coords)
        for i in 1:num_measures
            phi_col = phi[:, i, l]
            b[:, i, l] = p ./ phi_col
        end
    end

    if target !== nothing
        g = (p .- target) .* p
        for l in (iters-1):-1:2
            for m in 1:num_measures
                w[m] = w[m] + dot(log.(phi[:, m, l]), g)
                u = coords[m] .* g .- r[:, m]
                v = phi[:, m, l]
                b_m = b[:, m, l-1]
                p_m = measures[:, m]
                x = k * (u ./ v)
                y = p_m ./ ((k * b_m) .^ 2)
                result = (-k' * (x .* y)) .* b_m
                r[:, m] = result
            end
            g = vec(sum(r, dims=2))
        end
    else
        w = nothing
    end

    return p, w
end


"""
    sinkhorn_barycenter(coords, measures, target, cost, epsilon; iters=256)

Compute the Wasserstein barycenter of `measures` with weights `coords`
using the Sinkhorn algorithm with regularization `epsilon`.
"""
function sinkhorn_barycenter(coords, measures, target, cost, epsilon; iters=256)
    p, _ = sinkhorn_differentiate(coords, measures, target, cost, epsilon, iters)
    return p
end


"""
    barycentric_loss(coordinates, measures, target, cost, epsilon)

Compute the barycenter of `measures` with coordinates given by the
logarithmic change of variable applied to `coordinates`, and return
the squared Euclidean loss between that barycenter and `target`.
"""
function barycentric_loss(coordinates, measures, target, cost, epsilon)
    bar, _ = sinkhorn_differentiate(
        logarithmic_change_of_variable(coordinates),
        measures, target, cost, epsilon, 1024
    )
    return sqeuc_loss(bar, target)
end


"""
    loss_gradient(coords, measures, cost, target, epsilon)

Return the gradient of the barycentric loss w.r.t. coordinates,
as computed by the Bonneel algorithm (the `w` component of `sinkhorn_differentiate`).
"""
function loss_gradient(coords, measures, cost, target, epsilon)
    _, w = sinkhorn_differentiate(
        logarithmic_change_of_variable(coords),
        measures, target, cost, epsilon, 2048
    )
    return w
end


"""
    build_geodesic(measures, cost; epsilon=0.1, steps=10, iters=2048)

Given a pair of measures and a cost matrix, build a matrix whose columns are
points along the geodesic path from the first measure to the second.

`measures` must have exactly 2 columns.
"""
function build_geodesic(measures, cost; epsilon=0.1, steps=10, iters=2048)
    num_nodes, measure_count = size(measures)
    @assert measure_count == 2
    barycenters = zeros(num_nodes, steps + 1)
    for i in 0:steps
        coordinates = [1 - i / steps, i / steps]
        barycenters[:, i+1] = sinkhorn_barycenter(
            coordinates, measures, nothing, cost, epsilon; iters=iters
        )
    end
    return barycenters
end


"""
    simplex_regression(measures, target, cost, epsilon)

Given a family of `measures`, a `target` histogram, a `cost` matrix, and a
regularization parameter `epsilon`, find the barycentric coordinates that best
approximate `target` w.r.t. the reference measures via L-BFGS optimization.

Requires Optim.jl (`using Optim` must be available in the calling scope).
"""
function simplex_regression(measures, target, cost, epsilon)
    num_measures = size(measures, 2)
    x0 = fill(1.0 / num_measures, num_measures)
    f(coords) = barycentric_loss(coords, measures, target, cost, epsilon)
    function g!(G, coords)
        grad = loss_gradient(coords, measures, cost, target, epsilon)
        copyto!(G, grad)
    end
    result = Optim.optimize(f, g!, x0, Optim.LBFGS())
    return logarithmic_change_of_variable(Optim.minimizer(result))
end
