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

@testset "find_q_fast / project_by_newton_fast" begin
    proj_fast = GraphTransportation.project_by_newton_fast
    proj_old  = (x, y, z) -> begin
        v = GraphTransportation.project_by_newton(x, y, z)
        (v[1], v[2], v[3])
    end

    on_boundary(a, b, c) = abs(c - sqrt(max(a, 0.0) * max(b, 0.0)))

    # Stationarity residual: f(t) = z(1-t⁴) + t((x-2y)t² + (2x-y)) with t = √q = (b/a)^(1/2) ...
    # more directly: the projection normal should be parallel to ∇g = (-b, -a, 2c) at the projected pt
    function optimality_residual(x, y, z, a, b, c)
        # (x-a, y-b, z-c) must be proportional to (-b, -a, 2c)
        # cross-product magnitude (should be zero)
        r1 = (x - a) * (-a)  - (y - b) * (-b)   # i-j component of cross
        r2 = (y - b) * 2*c   - (z - c) * (-a)   # j-k component
        r3 = (x - a) * 2*c   - (z - c) * (-b)   # i-k component
        sqrt(r1^2 + r2^2 + r3^2) / (sqrt((x-a)^2+(y-b)^2+(z-c)^2) + 1e-30)
    end

    test_cases = [
        (1.0, 1.0, 2.0),    # symmetric: exact answer is (4/3, 4/3, 4/3)
        (4.0, 1.0, 3.0),    # asymmetric, D large
        (0.5, 2.0, 2.0),    # y > x
        (10.0, 0.1, 2.0),   # extreme ratio
        (100.0, 100.0, 200.0),  # large values
        (0.1, 0.1, 0.5),    # small values
        (2.0, 3.0, 4.0),
    ]

    @testset "matches old method" begin
        for (x, y, z) in test_cases
            @assert x + y > 0 && z > sqrt(x*y)  "not an exterior test case"
            a_old, b_old, c_old = proj_old(x, y, z)
            a_new, b_new, c_new = proj_fast(x, y, z)
            @test isapprox(a_old, a_new, atol=1e-6)
            @test isapprox(b_old, b_new, atol=1e-6)
            @test isapprox(c_old, c_new, atol=1e-6)
        end
    end

    @testset "projected point lies on K boundary" begin
        for (x, y, z) in test_cases
            a, b, c = proj_fast(x, y, z)
            @test on_boundary(a, b, c) < 1e-8
        end
    end

    @testset "optimality condition" begin
        for (x, y, z) in test_cases
            a, b, c = proj_fast(x, y, z)
            @test optimality_residual(x, y, z, a, b, c) < 1e-6
        end
    end

    @testset "known exact case (1,1,2) -> (4/3, 4/3, 4/3)" begin
        a, b, c = proj_fast(1.0, 1.0, 2.0)
        @test a ≈ 4/3  atol=1e-10
        @test b ≈ 4/3  atol=1e-10
        @test c ≈ 4/3  atol=1e-10
    end
end

@testset "projection_by_newton" begin
    proj = GraphTransportation.projection_by_newton

    # Returns the exact projected point: p + 0.25*q^2 = 0
    on_boundary(p, q) = abs(p + 0.25 * q^2)

    # Stationarity: residual of the depressed cubic that defines the projection
    cubic_residual(x, y, q) = abs(q^3 + 4*(2 + x)*q - 8*y)

    # Interior point: returned unchanged
    @testset "interior unchanged" begin
        for (x, y) in [(-1.0, 0.0), (-2.0, 1.0), (-5.0, 3.0)]
            @assert x + 0.25*y^2 <= 0 "test point not interior"
            p, q = proj(x, y)
            @test p == x && q == y
        end
    end

    # Boundary point: returned unchanged
    @testset "boundary unchanged" begin
        for q0 in [-2.0, 0.0, 1.5, 3.0]
            x, y = -0.25*q0^2, q0
            @assert abs(x + 0.25*y^2) < 1e-15
            p, q = proj(x, y)
            @test p == x && q == y
        end
    end

    # y = 0, x > 0: unique minimiser of distance is q = 0, so projects to (0, 0)
    @testset "y=0 projects to origin" begin
        for x in [0.1, 1.0, 5.0]
            p, q = proj(x, 0.0)
            @test p ≈ 0.0  atol=1e-10
            @test q ≈ 0.0  atol=1e-10
        end
    end

    # General exterior points (D >= 0): on-boundary and stationarity
    @testset "exterior D >= 0" begin
        for (x, y) in [(1.0, 2.0), (0.5, 1.0), (2.0, -3.0), (0.1, 0.5), (3.0, 4.0)]
            @assert x + 0.25*y^2 > 0 "test point not exterior"
            p, q = proj(x, y)
            @test on_boundary(p, q)    < 1e-10
            @test cubic_residual(x, y, q) < 1e-8
        end
    end

    # Three-root case (D < 0): requires x << -2 with |y| just large enough to be exterior
    # For x = -9, y = 6.1: D ≈ -218, confirmed exterior (-9 + 0.25*6.1^2 ≈ 0.30 > 0)
    @testset "exterior D < 0 (three-root case)" begin
        for (x, y) in [(-9.0, 6.1), (-9.0, -6.1), (-12.0, 7.5)]
            @assert x + 0.25*y^2 > 0 "test point not exterior"
            a = 4*(2 + x)
            b = -8*y
            D = (a/3)^3 + (b/2)^2
            @assert D < 0 "D >= 0, not a three-root case"
            p, q = proj(x, y)
            @test on_boundary(p, q)        < 1e-10
            @test cubic_residual(x, y, q) < 1e-8
            # confirm it is a local minimum by checking nearby points on the parabola
            dist(r) = (-0.25*r^2 - x)^2 + (r - y)^2
            @test all(dist(q + δ) >= dist(q) for δ in (-0.01, 0.01))
        end
    end
end
