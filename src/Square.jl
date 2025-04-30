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

function single_run(N=128, ε=0.1)
    Q = [0. 0.5 0. 0.5;
         0.5 0. 0.5 0.;
         0. 0.5 0. 0.5;
         0.5 0. 0.5 0.]
    μ = [4.; 0; 0.; 0.]
    ν = [0.; 4; 0.; 0.]
    a = [-1; 1/3; 1/3; 1/3;]
    b = [1/3; -1; 1/3; 1/3;]
    γ, d = BBD(Q, μ + ε * a, ν + ε * b, N, verbose=true)
    return (γ, d)

end

function export_vars()
    Q = [0. 0.5 0. 0.5;
         0.5 0. 0.5 0.;
         0. 0.5 0. 0.5;
         0.5 0. 0.5 0.]
    μ = [4.; 0; 0.; 0.]
    ν = [0.; 4; 0.; 0.]
    N = 2^7
    return (Q, μ, ν, N)
end


function negativity_extractor(S, T, ε=0.02)
    Q = [0. 0.5 0. 0.5;
         0.5 0. 0.5 0.;
         0. 0.5 0. 0.5;
         0.5 0. 0.5 0.]
    μ = [4.; 0; 0.; 0.]
    ν = [0.; 4; 0.; 0.]
    a = [-1; 1/3; 1/3; 1/3;]
    b = [1/3; -1; 1/3; 1/3;]
    arr = zeros(S, T+1)
    for i=1:S, (j, h) in enumerate(ε:ε:(T * ε))
        println("h = $(2.0^-i); ε = $(h)")
        c, d = BBD(Q, μ + h * a, ν + h * b, 2^i)
        arr[i, j] = !any(c.vector.ρ .< 0) ? 0 : 1
    end
    return arr
end
