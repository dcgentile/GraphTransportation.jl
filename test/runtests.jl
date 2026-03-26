using GraphTransportation
using Test
using SparseArrays
using LinearAlgebra

include("inclusion_helpers.jl")

@testset "discrete_transport / transport_cost" begin
    Q = [0.0 1.0; 1.0 0.0]
    a = [2.0, 0.0]
    b = [0.0, 2.0]
    geo  = discrete_transport(Q, a, b; N=100)
    dist = sqrt(action(geo))
    @test dist > 0
    @test isfinite(dist)
    @test transport_cost(Q, a, b; N=100) ≈ dist  atol=1e-6
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

@testset "graph_gradient / graph_divergence" begin
    Q = [0    1/2  0    1/2;
         1/3  0    1/3  1/3;
         0    1    0    0;
         1/2  1/2  0    0]
    f = [1, 2, 3, 4]
    ∇f_expected = [0.  -1.  0.  -3.;
                   1.   0. -1.  -2.;
                   0.   1.  0.   0.;
                   3.   2.  0.   0.]
    @test graph_gradient(Q, f) ≈ ∇f_expected

    m = [0  -1   0  -3;
         1   0  -1  -2;
         0   1   0   0;
         3   2   0   0]
    div_expected = [2, 2/3, -1, -5/2]
    @test graph_divergence(Q, m) ≈ div_expected
end

@testset "prox_Astar! (parabola projection)" begin
    # Generate random points outside the parabola x + y²/4 = 0 by perturbing
    # boundary points along the outward normal, then check projection recovers them.
    γ(t) = [-0.25 * t^2; t]
    N_pts(t) = [1, 0.5 * t]
    M = 2
    T = zeros(M, M, M)
    S = zeros(M, M, M)
    Tproj = zeros(M, M, M)
    Sproj = zeros(M, M, M)
    for i in eachindex(T)
        s = 0.5  # fixed to avoid randomness in CI
        p = γ(s)
        λ = 0.3
        v = p .+ (λ * N_pts(s))
        Tproj[i] = p[1]
        Sproj[i] = p[2]
        T[i] = v[1]
        S[i] = v[2]
    end
    GraphTransportation.prox_Astar!(T, S)
    @test T ≈ Tproj  atol=1e-8
    @test S ≈ Sproj  atol=1e-8
end

@testset "proximal_IJpm_star" begin
    Q        = [0. 1.; 1. 0.]
    ρ        = [1. 0.; 0. 1.]
    ρ_minus  = [1. 2.; 4. 5.;;; 2. 3.; 5. 6.]
    ρ_plus   = [7. 8.; 10. 11.;;; 8. 9.; 11. 12.]
    # hand-computed projections via Moreau's identity
    ρ_proj       = [3. 5/2; 4. 9/2]
    ρ_plus_proj  = [3. 3.; 4. 4.;;; 5/2 5/2; 9/2 9/2]
    ρ_minus_proj = [3. 5/2; 4. 9/2;;; 3. 5/2; 4. 9/2]

    @testset "non-mutating" begin
        ρ_hat, ρ_minus_hat, ρ_plus_hat =
            GraphTransportation.proximal_IJpm_star(ρ, ρ_minus, ρ_plus, Q)
        @test ρ_hat       ≈ ρ       .- ρ_proj
        @test ρ_minus_hat ≈ ρ_minus .- ρ_minus_proj
        @test ρ_plus_hat  ≈ ρ_plus  .- ρ_plus_proj
        @test is_in_JPM(ρ .- ρ_hat, ρ_minus .- ρ_minus_hat, ρ_plus .- ρ_plus_hat)
    end

    @testset "mutating matches non-mutating" begin
        ρ2       = copy(ρ)
        ρ_m2     = copy(ρ_minus)
        ρ_p2     = copy(ρ_plus)
        ρ_hat, ρ_minus_hat, ρ_plus_hat =
            GraphTransportation.proximal_IJpm_star(ρ2, ρ_m2, ρ_p2, Q)
        GraphTransportation.proximal_IJpm_star!(ρ2, ρ_m2, ρ_p2, Q)
        @test ρ2   ≈ ρ_hat
        @test ρ_m2 ≈ ρ_minus_hat
        @test ρ_p2 ≈ ρ_plus_hat
    end
end

@testset "prox_IJavg_star" begin
    N    = 3
    ρ_A  = [1.; 0.; 0.]
    ρ_B  = [0.; 0.; 1.]
    ρ    = [1    0    0;
            1/2  1/2  0;
            0    1/2  1/2;
            0    0    1]
    ρ_bar = [1.  0.  0.;
             0.  1.  0.;
             0.  0.  1.]
    Λ    = [17/70  -2/7   3/70;
            -3/14   3/7  -3/14;
             3/70  -2/7  17/70]
    ρ_pr = [0      0      0;
            -1/70  -5/70   6/70;
             6/70  -5/70  -1/70;
             0      0      0]
    M = GraphTransportation.form_avg_system(N)

    @testset "non-mutating" begin
        ρ_pr_c, ρ_bar_pr_c = GraphTransportation.prox_IJavg_star(copy(ρ), copy(ρ_bar), ρ_A, ρ_B, M)
        @test ρ_pr_c    ≈ ρ_pr  atol=1e-10
        @test ρ_bar_pr_c ≈ Λ    atol=1e-10
    end

    @testset "mutating" begin
        ρ2    = copy(ρ)
        ρ_bar2 = copy(ρ_bar)
        GraphTransportation.prox_IJavg_star!(ρ2, ρ_bar2, ρ_A, ρ_B, M)
        @test ρ2    ≈ ρ_pr  atol=1e-10
        @test ρ_bar2 ≈ Λ    atol=1e-10
    end
end

@testset "ContinuityEnforcer" begin
    @testset "form_ceh_system" begin
        # Hand-computed LU for a 2-node graph with N=6 time steps
        N = 6
        Q = [0 1; 1 0]
        A_hand = [-37  1  36   0   0   0   0   0   0   0   0   0  1;
                    1 -37   0  36   0   0   0   0   0   0   0   0  1;
                   36   0 -73   1  36   0   0   0   0   0   0   0  1;
                    0  36   1 -73   0  36   0   0   0   0   0   0  1;
                    0   0  36   0 -73   1  36   0   0   0   0   0  1;
                    0   0   0  36   1 -73   0  36   0   0   0   0  1;
                    0   0   0   0  36   0 -73   1  36   0   0   0  1;
                    0   0   0   0   0  36   1 -73   0  36   0   0  1;
                    0   0   0   0   0   0  36   0 -73   1  36   0  1;
                    0   0   0   0   0   0   0  36   1 -73   0  36  1;
                    0   0   0   0   0   0   0   0  36   0 -37   1  1;
                    0   0   0   0   0   0   0   0   0  36   1 -37  1;
                    1   1   1   1   1   1   1   1   1   1   1   1  0]
        ref = lu(sparse(A_hand))
        sys = GraphTransportation.form_ceh_system(Q, N)
        @test isapprox(ref.L * ref.U, sys.L * sys.U, atol=1e-10)
    end

    @testset "form_b" begin
        ρ_A = [1; 0]
        ρ_B = [0; 1]
        Q   = [0 1; 1 0]
        ρ   = [1 0; 0.75 0.25; 0.25 0.75; 0 1]
        m   = permutedims(cat([0 0.5; -0.5 0], [0 0.5; -0.5 0], [0 0.5; -0.5 0], dims=3), (3, 1, 2))
        target = -1 * [-5/4, 5/4, -2, 2, -5/4, 5/4, 0]
        @test isapprox(GraphTransportation.form_b(ρ_A, ρ_B, ρ, m, Q), target)
    end

    @testset "proj_CE" begin
        ρ_A = [1; 0]
        ρ_B = [0; 1]
        Q   = [0 1; 1 0]
        v   = [0.5; 0.5]
        ρ   = [1 0; 0.75 0.25; 0.25 0.75; 0 1]
        m   = permutedims(cat([0 0.5; -0.5 0], [0 0.5; -0.5 0], [0 0.5; -0.5 0], dims=3), (3, 1, 2))
        ∇φ  = [0 0; -3/116 3/116; 3/116 -3/116; 0 0]
        ρ_pr = ρ + 3 * ∇φ
        m_pr = m + permutedims(cat([0 -86/58; 86/58 0],
                                   [0 -89/58; 89/58 0],
                                   [0 -86/58; 86/58 0], dims=3), (3, 1, 2))

        @testset "non-mutating" begin
            ρ_hat, m_hat = GraphTransportation.proj_CE(ρ, m, ρ_A, ρ_B, Q)
            @test isapprox(ρ_hat, ρ_pr)
            @test isapprox(m_hat, m_pr)
            @test is_in_CE_weakly(ρ_hat, m_hat, Q, v)
        end

        @testset "mutating" begin
            ρ2 = copy(ρ)
            m2 = copy(m)
            GraphTransportation.proj_CE!(ρ2, m2, ρ_A, ρ_B, Q)
            @test isapprox(ρ2, ρ_pr)
            @test isapprox(m2, m_pr)
            @test is_in_CE_weakly(ρ2, m2, Q, v)
        end
    end
end

@testset "project_IJeq" begin
    ρ      = [1/3  2/3  1;  1/3  1/6  0;  1/3  1/6  0]
    q      = [1/2  3/4  1;  1/2  1/4  0;  0    0    0]
    answer = [5/12  17/24  1;  5/12  5/24  0;  1/6  1/12  0]
    ρ_pr, q_pr = GraphTransportation.project_IJeq(ρ, q)
    @test isapprox(ρ_pr, answer)
    @test isapprox(q_pr, answer)
    @test is_in_JEq(ρ_pr, q_pr)
end
