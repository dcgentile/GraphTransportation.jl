function project_IJeq(ρ::AbstractArray, q::AbstractArray)
    """
    compute the proximal mapping of IJ_eq in place

    arguments
    ρ, ρ_bar ∈ V^{0}_{n, h}, i.e, they are matrices of size 1/h × n, where h is the step size and n the number of nodes

    given ρ and q, this function returns their vector average
    """
	x = 0.5 .* (ρ .+ q)
    return (x, x)
end

function project_IJeq!(ρ::AbstractArray, q::AbstractArray)
    """
    compute the proximal mapping of IJ_eq in place

    arguments
    ρ, ρ_bar ∈ V^{0}_{n, h}, i.e, they are matrices of size 1/h × n, where h is the step size and n the number of nodes

    given ρ and q, this function returns their vector average
    """
	ρ .= q.= 0.5 .* (ρ .+ q)
end
