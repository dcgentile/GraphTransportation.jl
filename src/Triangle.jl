include("EarthMover.jl")

Q = [0. 0.5 0.5; 0.5 0. 0.5; 0.5 0.5 0.]
μ = [3.; 0; 0.]
ν = [0.; 3; 0.]

for i=2:9
    N = 2^i
    γ, d = BBD(Q, μ, ν, N)
    println("\nApproximated distance for h = 2^$(-i): d = $(d)")
    is_in_CE_weakly(γ.vector.ρ, γ.vector.m, γ.cache.Q, γ.cache.π)
end
