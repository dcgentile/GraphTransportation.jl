include("ErbarVector.jl")
include("galerkin/Chambolle.jl")
include("tests/Inclusion.jl")

function BBD(Q::AbstractMatrix,
             μ::AbstractVector,
             ν::AbstractVector,
             N=100;
             σ=0.5,
             τ=0.5,
             maxiters=2^16,
             tol=1e-10,
             verbose=false
             )
    geodesic = chambolle_pock_me(Q, μ, ν, N, maxiters=maxiters, σ=σ, τ=τ, tol=tol, verbose=verbose)
    return (geodesic, sqrt(action(geodesic)))
end


