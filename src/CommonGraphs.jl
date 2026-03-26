"""
    cube_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on the 3-cube (8 nodes, 12 edges).
Returns `(Q, π)` where `Q` is the transition matrix and `π` the stationary distribution.
"""
function cube_markov_chain()
    edge_list = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 6), (6, 7), (7, 8), (8, 5)
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
end

"""
    triangle_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on the 3-cycle (triangle).
"""
function triangle_markov_chain()
    edge_list = [(1,2), (2, 3), (3, 1)]
    Q, sstate = markov_chain_from_edge_list(edge_list)
    
    return (Q, sstate)
end

"""
    triangle_with_tail_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on a 3-cycle with one pendant
edge (node 4 attached to node 3).
"""
function triangle_with_tail_markov_chain()
    edge_list = [(1,2), (2, 3), (3, 1), (3, 4)]
    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
end

"""
    triangular_prism_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on the triangular prism graph
(6 nodes: two triangular faces connected by three edges).
"""
function triangular_prism_markov_chain()
    edge_list = [
        (1,2), (2, 3), (3, 1),
        (1,4), (2, 5), (3, 6),
        (4,5), (5, 6), (6, 4)
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)
    return (Q, sstate)
end


"""
    square_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on the 4-cycle (square).
"""
function square_markov_chain()
    edge_list = [(1,2), (2, 3), (3, 4), (4, 1)]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
end

"""
    T_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on the T-shaped graph
(path 1–2–3 with a branch 2–4).
"""
function T_markov_chain()
    edge_list = [
        (1,2), (2, 3), (2, 4),
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
	
end

"""
    double_T_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on the double-T graph
(8 nodes: two T-shaped graphs — nodes 1–4 and nodes 5–8 — connected at
their corresponding vertices: 1↔5, 2↔6, 3↔7, 4↔8).
"""
function double_T_markov_chain()
    edge_list = [
        (1,2), (2, 3), (2, 4),
        (1,5), (2, 6), (3, 7), (4, 8),
        (5,6), (6, 7), (6, 8),
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
	
end

"""
    grid_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on a 9-node wheel-like graph:
an 8-cycle (nodes 1–8) with a central hub (node 9) connected to the even nodes.
"""
function grid_markov_chain()
    edge_list = [
        (1, 2), (2, 3), (3, 4),
        (3, 4), (4, 5), (5, 6),
        (6, 7), (7, 8), (8, 1),
        (9, 2), (9, 4), (9, 6), (9, 8)
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
	
end

"""
    weighted_hypercube_markov_chain() -> (Q, π)

Construct a randomly weighted random walk Markov chain on the 4-cube (16 nodes,
32 edges).  Edge weights are drawn uniformly from `{1, …, 10}` and symmetrised,
so the stationary distribution is non-uniform.  Results vary across calls.
"""
function weighted_hypercube_markov_chain()
    edge_list = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 6), (6, 7), (7, 8), (8, 5),

        (1,9), (2, 10), (3, 11), (4, 12),
        (5, 13), (6, 14), (7, 15), (8, 16),

        (9, 10), (10, 11), (11, 12), (12, 9),
        (9, 13), (10, 14), (11, 15), (12, 16),
        (13, 14), (14, 15), (15, 16), (16, 13)
    ]

    A = zeros(16,16)
    for e in edge_list
        i,j = e
        A[i,j] = A[j,i] = 1
    end

    M = rand(1:10, 16, 16)
    M = M + M'
    W = A .* M

    Q, π = markov_chain_from_weight_matrix(W)

    return (Q, π)


	
end

"""
    hypercube_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on the 4-cube (16 nodes,
32 edges).  All edges have equal weight, so the stationary distribution is
uniform.
"""
function hypercube_markov_chain()
    edge_list = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 6), (6, 7), (7, 8), (8, 5),

        (1,9), (2, 10), (3, 11), (4, 12),
        (5, 13), (6, 14), (7, 15), (8, 16),

        (9, 10), (10, 11), (11, 12), (12, 9),
        (9, 13), (10, 14), (11, 15), (12, 16),
        (13, 14), (14, 15), (15, 16), (16, 13)
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
	
end

"""
    grid_markov_chain(n) -> (Q, π)

Construct the uniform random walk Markov chain on the `n × n` grid graph
(`n²` nodes with nearest-neighbour edges).
"""
function grid_markov_chain(n)
    E = Tuple{Int,Int}[]
    for i in 1:n^2
        i % n != 0 && (push!(E, (i, i+1)); push!(E, (i+1, i)))
        i <= n*(n-1) && (push!(E, (i, i+n)); push!(E, (i+n, i)))
    end
    return markov_chain_from_edge_list(E)
end

"""
    ma_house_markov_chain() -> (Q, π)

Construct the uniform random walk Markov chain on the Massachusetts state house
district adjacency graph.  Nodes are legislative districts; edges connect
districts that share a non-trivial boundary (point adjacencies are excluded).

Reads district geometries from the bundled shapefile at
`src/experiments/HOUSE2021/HOUSE2021_POLY.shp`.  Requires `LibGEOS` and
`Shapefile` to be loaded.
"""
function ma_house_markov_chain()
    GI = LibGEOS.GeoInterface

    shp_file = joinpath(@__DIR__, "experiments", "HOUSE2021", "HOUSE2021_POLY.shp")
    table = Shapefile.Table(shp_file)
    geoms = Shapefile.shapes(table)
    N = length(geoms)

    geos_geoms = Vector{LibGEOS.MultiPolygon}(undef, N)
    for i in 1:N
        geos_geoms[i] = GI.convert(LibGEOS.MultiPolygon, geoms[i])
    end

    edges_list = Tuple{Int,Int}[]
    prepared = [LibGEOS.prepareGeom(g) for g in geos_geoms]
    for i in 1:N
        for j in (i+1):N
            if LibGEOS.intersects(prepared[i], geos_geoms[j])
                intersection = LibGEOS.intersection(geos_geoms[i], geos_geoms[j])
                if !(intersection isa Union{LibGEOS.Point, LibGEOS.MultiPoint})
                    push!(edges_list, (i, j))
                end
            end
        end
    end

    A = zeros(N, N)
    for (i, j) in edges_list
        A[i, j] = 1
        A[j, i] = 1
    end
    Q, sstate = markov_chain_from_weight_matrix(A)
    return (Q, sstate)
end
