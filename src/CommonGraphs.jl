include("utils.jl")

function cube_markov_chain()
    edge_list = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 6), (6, 7), (7, 8), (8, 5)
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
end

function triangle_markov_chain()
    edge_list = [(1,2), (2, 3), (3, 1)]
    Q, sstate = markov_chain_from_edge_list(edge_list)
    
    return (Q, sstate)
end

function triangle_with_tail_markov_chain()
    edge_list = [(1,2), (2, 3), (3, 1), (3, 4)]
    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
end

function triangular_prism_markov_chain()
    edge_list = [
        (1,2), (2, 3), (3, 1),
        (1,4), (2, 5), (3, 6),
        (4,5), (5, 6), (6, 4)
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)
    return (Q, sstate)
end


function square_markov_chain()
    edge_list = [(1,2), (2, 3), (3, 4), (4, 1)]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
end

function T_markov_chain()
    edge_list = [
        (1,2), (2, 3), (2, 4),
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
	
end

function double_T_markov_chain()
    edge_list = [
        (1,2), (2, 3), (2, 4),
        (1,5), (2, 6), (3, 7), (4, 8),
        (5,6), (6, 7), (6, 8),
    ]

    Q, sstate = markov_chain_from_edge_list(edge_list)

    return (Q, sstate)
	
end

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
