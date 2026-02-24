struct DiscreteMetricSpace
    N::Int32
    E::AbstractArray
    Q::AbstractArray
    π::AbstractArray

    function DiscreteMetricSpace(A::Matrix)
	    # initialize with an adjacency matrix
    end

    function DiscreteMetricSpace(W::Matrix)
        # initialize with a weighted adjacency matrix
    end

    function DiscreteMetricSpace(Q::Matrix)
	    # initalize with a Markov kernel 
    end

    function DiscreteMetricSpace(Q::Matrix, steady_state::Vector)
	    # initalize with a Markov kernel and prescribed steady state
    end
    
end
