include("EarthMover.jl")

Q = [0. 0.5 0. 0.5; 0.5 0. 0.5 0.; 0. 0.5 0. 0.5; 0.5 0. 0.5 0.]
μ = [4.; 0; 0.; 0.]
ν = [0.; 4; 0.; 0.]

for i=2:9
    N = 2^i
    γ, d = BBD(Q, μ, ν, N)
    println("\nApproximated distance for h = 2^$(-i): d = $(d)")
    is_in_CE(γ.vector.ρ, γ.vector.m, γ.cache.Q, γ.cache.π)
end
