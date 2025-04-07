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

function single_run(n)
    Q = [0. 0.5 0. 0.5;
         0.5 0. 0.5 0.;
         0. 0.5 0. 0.5;
         0.5 0. 0.5 0.]
    μ = [4.; 0; 0.; 0.]
    ν = [0.; 4; 0.; 0.]
    N = 2^n
    #Q = convert(Array{BigFloat}, Q)
    #μ = convert(Array{BigFloat}, μ)
    #ν = convert(Array{BigFloat}, ν)
    γ, d = BBD(Q, μ, ν, N)
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
