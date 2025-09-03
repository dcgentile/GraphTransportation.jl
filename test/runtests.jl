using GraphTransportation
using Test

@testset "GraphTransportation.jl" begin
    Q = [0. 1.; 1. 0.];
    a = [2.0; 0];
    b = [0.; 2];
    N = 100;
    v, dist = BBD(Q, a, b, N);
    println("Distance between Dirac masses on a 2 point graph: $(dist)")
end
