include("EarthMover.jl")

function triangle_transport(N)
    Q = [0. 0.5 0.5; 0.5 0. 0.5; 0.5 0.5 0.]
    μ = [3.; 0; 0.]
    ν = [0.; 3; 0.]

    γ, d = BBD(Q, μ, ν, N)
    return (γ, d)
end
