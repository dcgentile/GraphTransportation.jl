"""
    project_IJeq(ρ, q) -> (x, x)

Project `(ρ, q)` onto `J_eq = {(ρ, q) : ρ = q}` by returning their
element-wise average: both output arrays equal `0.5·(ρ + q)`.

`ρ` and `q` must be matrices of size `N × V`.
"""
function project_IJeq(ρ::AbstractArray, q::AbstractArray)
	x = 0.5 .* (ρ .+ q)
    return (x, x)
end

"""
    project_IJeq!(ρ, q) -> (ρ, q)

In-place variant of `project_IJeq`: sets both `ρ` and `q` to
`0.5·(ρ + q)` without allocating.
"""
function project_IJeq!(ρ::AbstractArray, q::AbstractArray)
	ρ .= q .= 0.5 .* (ρ .+ q)
    return (ρ, q)
end
