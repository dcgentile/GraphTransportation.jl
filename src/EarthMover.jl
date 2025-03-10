include("ErbarVector.jl")
include("galerkin/Chambolle.jl")

function BBD(Q::AbstractMatrix,
             μ::AbstractVector,
             ν::AbstractVector,
             N=100;
             tol=1e-5
             )
    geodesic = chambolle_pock_me(Q, μ, ν, N, tol=tol)
    return (geodesic, sqrt(action(geodesic)))
end


#Q = [0. 1.; 1. 0.]
#μ = [2.; 0.]
#ν = [0.; 2.]

Q = [0. 1. 1.; 1. 0. 1.; 1. 1. 0.]
μ = [3.; 0.; 0]
ν = [0.; 3.; 0]
dist = BBD(Q, μ, ν, 100, tol=1e-8)

#println(sqrt(dist))
