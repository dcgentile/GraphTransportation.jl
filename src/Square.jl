include("EarthMover.jl")

function run_experiment()
    Q = [0. 0.5 0. 0.5; 0.5 0. 0.5 0.; 0. 0.5 0. 0.5; 0.5 0. 0.5 0.]
    μ = [4.; 0; 0.; 0.]
    ν = [0.; 4; 0.; 0.]

    for i=2:9
        N = 2^i
        _, d = BBD(Q, μ, ν, N)
        println("\nApproximated distance for h = 2^$(-i): d = $(d)")
    end
    return BBD(Q, μ, ν, 100)
end
