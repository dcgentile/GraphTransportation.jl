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
    # Algorithm 1 of Bonneel, Peyré & Cuturi (2016). Index convention: the paper's
    # b^(0) = 1 is stored at slot 1 and its Sinkhorn iterations ℓ = 1..L at slots
    # 2..iters, so `iters` slots give L = iters - 1 iterations.
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
        # Reverse loop over every forward iteration, ℓ = L, ..., 1 (slots iters, ..., 2).
        # Starting one slot lower drops the top term of the sum: at small L that gives
        # a wrong gradient (wrong sign at L = 2), and it only becomes negligible once
        # the forward iterations have converged.
        for l in iters:-1:2
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
    barycentric_loss(α, measures, target, cost, epsilon; iters=256)

The regression objective `E_L(λ)` of Bonneel et al. (Eq. 12) with the squared Euclidean
loss, as a function of the *unconstrained* variable `α` through the softmax change of
variables `λ = softmax(α)` (`logarithmic_change_of_variable`). The same `iters` must be
used for the objective and its gradient (`loss_gradient`).
"""
function barycentric_loss(α, measures, target, cost, epsilon; iters=256)
    bar, _ = sinkhorn_differentiate(logarithmic_change_of_variable(α), measures, target, cost, epsilon, iters)
    return sqeuc_loss(bar, target)
end


"""
    loss_gradient(α, measures, cost, target, epsilon; iters=256)

Gradient of `barycentric_loss` with respect to the unconstrained variable `α`:
`sinkhorn_differentiate` returns `∇_λ E_L` (Algorithm 1's `w`), and the softmax
change of variables `λ = softmax(α)` contributes its Jacobian,
`∇_α E = λ ∘ (∇_λ E − ⟨λ, ∇_λ E⟩)`. Checked against finite differences in the tests.
"""
function loss_gradient(α, measures, cost, target, epsilon; iters=256)
    λ = logarithmic_change_of_variable(α)
    _, w = sinkhorn_differentiate(λ, measures, target, cost, epsilon, iters)
    return λ .* (w .- dot(λ, w))
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
    simplex_regression(measures, target, cost, epsilon; iters=256, α0=zeros(S), optim_options=Optim.Options())

Wasserstein barycentric coordinates of `target` with respect to the columns of
`measures` (Bonneel, Peyré & Cuturi 2016, §4.3): minimize `E_L(λ) = ½‖P^(L)(λ) − target‖²`
over the simplex by L-BFGS on `α` with `λ = softmax(α)`, using the gradient from
`sinkhorn_differentiate` with the softmax Jacobian applied. `α0 = 0` is the paper's
`λ0 = 1/S`. `iters` is the Sinkhorn iteration budget used for both the objective and the
gradient. Returns `λ̂ ∈ Δ`.
"""
function simplex_regression(measures, target, cost, epsilon; iters=256,
                            α0=zeros(size(measures, 2)), optim_options=Optim.Options())
    f(α)     = barycentric_loss(α, measures, target, cost, epsilon; iters=iters)
    g!(G, α) = copyto!(G, loss_gradient(α, measures, cost, target, epsilon; iters=iters))
    result = Optim.optimize(f, g!, α0, Optim.LBFGS(), optim_options)
    return logarithmic_change_of_variable(Optim.minimizer(result))
end
