using Shapefile, LibGEOS
using Statistics, LinearAlgebra, SparseArrays
"""
    random_geographic_concentration(adj_mat; weight=10, center=nothing)

Return a probability measure on the graph that is concentrated at `center` and its
neighbors, with all other nodes receiving weight 1 (before normalization).
"""
function random_geographic_concentration(adj_mat; weight=100, center=nothing)
    num_nodes = size(adj_mat, 1)
    if isnothing(center)
        center = rand(1:num_nodes)
    end
    measure = ones(num_nodes)
    measure[center] *= weight
    for index in 1:num_nodes
        adj_mat[center, index] == 1 && (measure[index] *= weight)
    end
    return measure ./ norm(measure, 1)
end

function load_usa_mc()
    function kernel_from_adjacency(adj)
        N, _ = size(adj)
        A = sparse(adj)
        nedges = nnz(A)
        degree_vector = A * ones(N)
        sstate = degree_vector / nedges
        Q = zeros(size(A))
        for i = 1:N, j = 1:N
            if A[i,j] != 0
                Q[i,j] = 1 / (sstate[i] * nedges)
            end
        end
        return Q, sstate
    end

    shapes = Shapefile.Handle("./data/states.shp").shapes
    n = length(shapes)

    adj = [touches(shapes[i], shapes[j]) || intersects(shapes[i], shapes[j])
           for i in 1:n, j in 1:n]

    geo_cx = Float64[]
    geo_cy = Float64[]
    for shape in shapes
        points = shape.points
        push!(geo_cx, mean(p.x for p in points))
        push!(geo_cy, mean(p.y for p in points))
    end

    Qusa, sstate = kernel_from_adjacency(adj - I(49))
    return Qusa, sstate, geo_cx, geo_cy
end

"""
    diffusion_distance(mchain, steady_state, i, j)

Compute the diffusion distance between nodes `i` and `j` given a (time-scaled)
Markov chain and its steady state.
"""
function diffusion_distance(mchain, steady_state, i, j)
    v = ((mchain[i, :] .- mchain[j, :]) .^ 2) ./ steady_state
    return sqrt(sum(v))
end


"""
    compute_graph_metric(adj_mat)

Return the all-pairs shortest-path distance matrix for the graph defined by `adj_mat`.
"""
function compute_graph_metric(adj_mat)
    n = size(adj_mat, 1)
    g = SimpleGraph(n)
    for i in 1:n, j in i+1:n
        adj_mat[i, j] != 0 && add_edge!(g, i, j)
    end
    dist_mat = zeros(n, n)
    for i in 1:n
        d = dijkstra_shortest_paths(g, i)
        dist_mat[i, :] = d.dists
    end
    return dist_mat
end


"""
    form_diffusion_map_from_graph(adj_mat, t)

Given an adjacency matrix and a time parameter `t`, compute the matrix of pairwise
diffusion distances under the random walk at time `t`.
"""
function form_diffusion_map_from_graph(adj_mat, t)
    mchain = adj_mat_to_markov_chain(Float64.(adj_mat))
    time_scaled_chain = mchain^t
    steady_state = find_markov_steady_state(mchain)
    dim = size(mchain, 1)
    D = zeros(dim, dim)
    for i in 1:dim, j in i+1:dim
        D[i, j] = diffusion_distance(time_scaled_chain, steady_state, i, j)
        D[j, i] = D[i, j]
    end
    return D
end
