using GraphTransportation
using GraphTransportation: geodesic_socp, barycenter_socp, analyze_socp, analyze_shooting, discrete_transport
using Test
using SparseArrays
using LinearAlgebra
using Random
using JuMP, Clarabel
using QuadGK
using ForwardDiff

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

@testset "MarkovGraph / graph_gradient / graph_divergence" begin
    graphs = [cube_markov_chain(), weighted_hypercube_markov_chain(), triangle_markov_chain()]

    @testset "adjoint identity ⟨φ, div m⟩_π = -⟨∇φ, m⟩_Q" begin
        Random.seed!(42)
        for (Q, π) in graphs
            G = MarkovGraph(Q, π)
            for _ in 1:10
                φ = randn(G.n)
                m = randn(length(G.E))

                lhs = dot(φ, graph_divergence(G, m) .* π)
                rhs = -dot(graph_gradient(G, φ) .* G.κ, m)

                @test lhs ≈ rhs atol=1e-12
            end
        end
    end

    @testset "reversibility check catches a broken chain" begin
        Q, π = cube_markov_chain()
        Q_broken = copy(Q)
        idx = findfirst(!=(0), Q_broken)
        Q_broken[idx] *= 2  # break Q(x,y)π(x) == Q(y,x)π(y)
        @test_throws AssertionError MarkovGraph(Q_broken, π)
    end

    @testset "κ agrees from both directions" begin
        Q, π = weighted_hypercube_markov_chain()
        G = MarkovGraph(Q, π)
        for (e, (x, y)) in enumerate(G.E)
            @test G.κ[e] ≈ Q[x, y] * π[x] atol=1e-12
            @test G.κ[e] ≈ Q[y, x] * π[y] atol=1e-12
        end
    end

    # Cross-validate the new sparse edge-vector grad/div against the pre-existing,
    # already-tested dense (V×V matrix) implementations, rather than only checking
    # internal self-consistency via the adjoint identity above.
    @testset "matches dense graph_gradient/graph_divergence" begin
        Random.seed!(7)
        for (Q, π) in graphs
            G = MarkovGraph(Q, π)
            for _ in 1:20
                φ = randn(G.n)
                ∇φ_dense = graph_gradient(Q, φ)
                ∇φ_sparse = graph_gradient(G, φ)
                for (e, (x, y)) in enumerate(G.E)
                    @test ∇φ_sparse[e] ≈ ∇φ_dense[x, y] atol=1e-12
                end

                m = randn(length(G.E))
                m_dense = zeros(G.n, G.n)
                for (e, (x, y)) in enumerate(G.E)
                    m_dense[x, y] = m[e]
                    m_dense[y, x] = -m[e]
                end
                @test graph_divergence(G, m) ≈ graph_divergence(Q, m_dense) atol=1e-10
            end
        end
    end

    # graph_divergence(G::MarkovGraph, ·) must also work when `m` holds JuMP variables
    # (as geodesic_socp requires), not just Float64s. Note: div's image is the
    # π-weighted-mean-zero subspace (⟨1, div m⟩_π = -⟨∇1, m⟩_Q = 0 for any m, since
    # ∇1 = 0), so `target` must be feasible for that reason, not chosen arbitrarily —
    # here we take it to be the divergence of a known ground-truth m.
    @testset "graph_divergence works with JuMP variables" begin
        Q, π = triangle_markov_chain()
        G = MarkovGraph(Q, π)
        m_truth = randn(length(G.E))
        target = graph_divergence(G, m_truth)

        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, m[1:length(G.E)])
        @constraint(model, graph_divergence(G, m) .== target)
        @objective(model, Min, sum(m .^ 2))
        optimize!(model)
        @test termination_status(model) == OPTIMAL
        @test graph_divergence(G, value.(m)) ≈ target atol=1e-6
    end
end

@testset "geodesic_socp vs two-node closed form" begin
    # Two-node graph, θ = geometric mean. Parameterize densities w.r.t. π=[0.5,0.5]
    # by r ∈ (-1,1): ρ(r) = [1-r, 1+r]. The closed-form distance is
    #   W(ρ(s), ρ(t)) = (1/√2) ∫_s^t (1-r²)^{-1/4} dr
    # and the geodesic satisfies ODE γ'(τ) = C·√2·((1-γ)(1+γ))^{1/4}, γ(0)=s, C=W(ρ(s),ρ(t)).
    # This is the same closed form and explicit-Euler reference used in the paper's
    # ODE-comparison experiment (src/experiments/ErbarODE.jl), which cross-validates the
    # existing Chambolle-Pock discrete_transport. This test gates geodesic_socp's RSOC
    # scaling/factor conventions against the same ground truth, independent of any
    # convention choice inside geodesic_socp itself.
    G = MarkovGraph([0.0 1.0; 1.0 0.0], [0.5, 0.5])

    ε_reg = 1e-4  # avoid the integrable-but-singular boundary values ±1
    s = -1.0 + ε_reg
    t =  1.0 - ε_reg
    ρA = [1.0 - s, 1.0 + s]
    ρB = [1.0 - t, 1.0 + t]

    W_ref(a, b) = quadgk(r -> (1 - r^2)^(-1/4), a, b)[1] / sqrt(2)
    C = W_ref(s, t)

    function ode_euler_interp(a, b, C; N=2000)
        h = 1.0 / N
        γ = Vector{Float64}(undef, N + 1)
        γ[1] = a
        for i in 1:N
            x = γ[i]
            γ[i+1] = clamp(x + h * C * sqrt(2) * max(0.0, (1 - x) * (1 + x))^(1/4), a, b)
        end
        return τ -> begin
            raw  = τ * N
            lo   = clamp(floor(Int, raw), 0, N - 1)
            frac = raw - lo
            γ[lo + 1] * (1 - frac) + γ[lo + 2] * frac
        end
    end
    ode_γ = ode_euler_interp(s, t, C)

    # Both the SOCP's own time discretization and the reference are O(h) accurate, so
    # the discrepancy between them should also be O(h); check this at increasing N
    # rather than pinning a single tolerance.
    prev_W_err = Inf
    prev_ρ_err = Inf
    for N in (10, 20, 50, 100)
        sol = geodesic_socp(G, ρA, ρB; N=N)
        @test sol.status == OPTIMAL

        W_err = abs(sqrt(sol.W2) - C)
        @test W_err < 2.0 / N   # O(h) with generous constant

        ts = range(0.0, 1.0, length=N + 1)
        ρ_err = maximum(abs(sol.ρ[2, i] - (1.0 + ode_γ(τ))) for (i, τ) in enumerate(ts))
        @test ρ_err < 2.0 / N

        @test W_err < prev_W_err + 1e-9   # error should not grow as N increases
        @test ρ_err < prev_ρ_err + 1e-9
        prev_W_err, prev_ρ_err = W_err, ρ_err
    end
end

@testset "geodesic_socp endpoint potentials: dual sign/scale calibration" begin
    # Before building anything on JuMP's duals,
    # pin down their sign and scale. `φ0`/`φ1` are defined as the gradient of W2 with
    # respect to each endpoint density in the π-weighted pairing, so they must match
    # (a) a central finite difference of `geodesic_socp`'s own W2 (tight — same
    # discretization on both sides) and (b) the two-node closed form's derivative,
    # d/ds W(ρ(s),ρ(t))² = -√2·W·(1-s²)^{-1/4}, with O(h) error.
    G = MarkovGraph([0.0 1.0; 1.0 0.0], [0.5, 0.5])
    s, t = -0.6, 0.7
    ρ(r) = [1.0 - r, 1.0 + r]           # δρ/dr = [-1, 1], so ⟨φ, δρ⟩_π = (φ[2] - φ[1])/2
    W_ref(a, b) = quadgk(r -> (1 - r^2)^(-1/4), a, b)[1] / sqrt(2)
    C = W_ref(s, t)
    dW2_ds = -sqrt(2) * C * (1 - s^2)^(-1/4)
    dW2_dt =  sqrt(2) * C * (1 - t^2)^(-1/4)

    ε = 1e-4
    for N in (5, 20, 100)
        sol = geodesic_socp(G, ρ(s), ρ(t); N=N)
        @test sol.status == OPTIMAL
        pair(φ) = (φ[2] - φ[1]) / 2

        fd_s = (geodesic_socp(G, ρ(s + ε), ρ(t); N=N).W2 - geodesic_socp(G, ρ(s - ε), ρ(t); N=N).W2) / (2ε)
        fd_t = (geodesic_socp(G, ρ(s), ρ(t + ε); N=N).W2 - geodesic_socp(G, ρ(s), ρ(t - ε); N=N).W2) / (2ε)
        @test pair(sol.φ0) ≈ fd_s rtol=1e-3      # sign: no flip of JuMP's dual needed
        @test pair(sol.φ1) ≈ fd_t rtol=1e-3
        @test abs(pair(sol.φ0) - dW2_ds) < 1.0 / N
        @test abs(pair(sol.φ1) - dW2_dt) < 1.0 / N
    end

    # The continuity-equation duals ψ_t are the half-step potentials and relate to the
    # momenta through the interval *midpoint* density: m_t = -(1/2h) θ(ρ̄_t) ∇(ψ_t/π).
    # This is the `θ(ρ̄)∘∇φ = m` calibration; it is exact (to solver
    # tolerance) only with ρ̄_t, which is why reading φ off these duals and pairing it
    # with θ(ρ0) looked "not a clean constant" in an earlier attempt.
    Q, π = triangle_markov_chain()
    G3 = MarkovGraph(Q, π)
    N = 4; h = 1.0 / N
    model = Model(Clarabel.Optimizer); set_silent(model)
    blk = GraphTransportation._geodesic_block!(model, G3, N, h, [2.0, 0.5, 0.5], [0.4, 0.4, 2.2])
    @objective(model, Min, h * sum(G3.κ[e] * blk.w[e, tt] for tt in 1:N, e in 1:length(G3.E)))
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    ρ_path = value.(blk.ρ); m_path = value.(blk.m)
    for tt in 1:N
        ψ = dual.(blk.c_cont[tt]) ./ G3.π
        ρ̄ = (ρ_path[:, tt] .+ ρ_path[:, tt+1]) ./ 2
        predicted = -(1 / (2h)) .* metric_tensor(G3, ρ̄) .* graph_gradient(G3, ψ)
        @test predicted ≈ m_path[:, tt] rtol=1e-3   # solver tolerance, not O(h)
    end
end

@testset "barycenter_socp endpoint potentials and KKT stationarity" begin
    # The per-block potentials returned by barycenter_socp (with the λ[i] weighting
    # divided out) must be the same as an independent geodesic_socp solve's, and the
    # joint program's stationarity in ν is exactly Σᵢ λᵢ φ1ᵢ = const on supp(ν)
    # Potentials are only defined up to an additive
    # constant, so compare gradients.
    Q, π = triangle_markov_chain()
    G = MarkovGraph(Q, π)
    refs = [[2.0, 0.5, 0.5], [0.5, 2.0, 0.5], [0.5, 0.5, 2.0]]
    λ = [0.5, 0.3, 0.2]
    ν, _, geos = barycenter_socp(G, refs, λ; N=3)
    @test all(ν .> 1e-3)   # fully supported, so stationarity has no slack term

    for (i, geo) in enumerate(geos)
        indep = geodesic_socp(G, refs[i], ν; N=3)
        @test graph_gradient(G, geo.φ1) ≈ graph_gradient(G, indep.φ1) rtol=1e-3   # solver tolerance
        @test graph_gradient(G, geo.φ0) ≈ graph_gradient(G, indep.φ0) rtol=1e-3
    end

    stationarity = sum(λ[i] .* geos[i].φ1 for i in eachindex(geos))
    scale = maximum(abs, graph_gradient(G, geos[1].φ1))
    @test maximum(abs, graph_gradient(G, stationarity)) < 1e-6 * scale
end

@testset "geodesic_socp vs Chambolle-Pock" begin
    # Cross-validate against the incumbent Chambolle-Pock solver on the 3-cycle,
    # 4-cycle, and 3x3 grid (Erbar Figs. 5-6 configurations). Unlike the exact two-node
    # two-node case, there's no closed form here, so - as with the Chambolle-Pock step-size
    # regression test above - we check convergence to a common value as N grows rather
    # than a fixed tolerance at small N, since both solvers carry their own O(h) time-
    # discretization error. Requires fixed-step Chambolle-Pock: with the former accelerated
    # accelerated default, this comparison would not converge (see the testset above).
    graphs = [
        ("3-cycle", triangle_markov_chain()),
        ("4-cycle", square_markov_chain()),
        ("3x3 grid", grid_markov_chain(3)),
    ]
    for (name, (Q, π)) in graphs
        @testset "$name" begin
            G = MarkovGraph(Q, π)
            rng = MersenneTwister(99)
            μ = (rand(rng, G.n) .+ 0.1); μ ./= dot(μ, π)
            ν = (rand(rng, G.n) .+ 0.1); ν ./= dot(ν, π)

            prev_relW = Inf
            prev_path_err = Inf
            for N in (10, 20, 50, 100)
                sol = geodesic_socp(G, μ, ν; N=N)
                geo = discrete_transport(Q, μ, ν; N=N, tol=1e-12, maxiters=2^20)
                W2_cp = action(geo)

                relW = abs(W2_cp - sol.W2) / sol.W2
                @test relW < 2.0 / N
                @test relW < prev_relW + 1e-9
                prev_relW = relW

                ρ_cp = permutedims(geo.vector.ρ)  # (N+1) × n -> n × (N+1)
                path_err = maximum(abs.(ρ_cp .- sol.ρ))
                @test path_err < 1.0 / N
                @test path_err < prev_path_err + 1e-9
                prev_path_err = path_err
            end
        end
    end
end

@testset "barycenter_socp vs geodesic_socp: p=2 sanity" begin
    # Bary({ν0,ν1}, (1-t,t)) must equal the geodesic point ν(t): with weights summing
    # to 1 over exactly two references, the barycenter SOCP and the geodesic SOCP solve
    # (mathematically) the same joint problem, so at grid times t=k/N they should agree
    # at solver tolerance - much tighter than ~1e-3 relative error
    # for the intrinsic-descent comparison (Figs. 9-10), since both are now convex solves.
    # Clarabel's achieved precision varies with the LAPACK/BLAS shipped by the Julia
    # version (observed max error ~1.3e-4 on 1.11/1.12/pre vs ~1.5e-3 on 1.10), so the
    # bound below carries margin above worst-case observed solver noise rather than
    # sitting at the tighter value achievable on newer Julia versions.
    Q, π = grid_markov_chain(3)
    G = MarkovGraph(Q, π)
    rng = MersenneTwister(1)
    ν0 = (rand(rng, G.n) .+ 0.1); ν0 ./= dot(ν0, π)
    ν1 = (rand(rng, G.n) .+ 0.1); ν1 ./= dot(ν1, π)

    N = 10
    sol = geodesic_socp(G, ν0, ν1; N=N)

    for k in 0:N
        t = k / N
        ν_bary = if t == 0.0
            ν0
        elseif t == 1.0
            ν1
        else
            first(barycenter_socp(G, [ν0, ν1], [1 - t, t]; N=N))
        end
        @test maximum(abs.(ν_bary .- sol.ρ[:, k+1])) < 3e-3
    end
end

@testset "barycenter_socp: symmetric sanity check" begin
    # Three references related by the triangle's cyclic symmetry, equal weights: by
    # symmetry the barycenter must be the uniform (w.r.t. π) density, and all three
    # reference-to-barycenter distances must be equal.
    Q, π = triangle_markov_chain()
    G = MarkovGraph(Q, π)
    refs = [[2.0, 0.5, 0.5], [0.5, 2.0, 0.5], [0.5, 0.5, 2.0]]

    ν, J, geos = barycenter_socp(G, refs, fill(1/3, 3); N=10)
    @test ν ≈ ones(3) atol=1e-6
    @test all(g -> g.status == OPTIMAL, geos)
    W2s = [g.W2 for g in geos]
    @test W2s[1] ≈ W2s[2] atol=1e-6
    @test W2s[2] ≈ W2s[3] atol=1e-6
    @test J ≈ sum(W2s) / 3 atol=1e-6

    @testset "λ[i]==0 drops that reference" begin
        ν2, J2, geos2 = barycenter_socp(G, refs, [0.5, 0.5, 0.0]; N=10)
        @test length(geos2) == 2
    end
end

@testset "analyze_socp: recovers barycentric coordinates" begin
    # Synthesize a barycenter with barycenter_socp,
    # then recover its coordinates with analyze_socp and check they match the synthesis
    # weights. Also cross-check against the existing (dissertation-validated)
    # Chambolle-Pock-based `analysis`, since analyze_socp reuses its exact Gram/QP
    # formulation (see solve_barycentric_coordinates_qp) - the two should agree
    # closely, not just each independently recover the truth.
    Q, π = triangle_markov_chain()
    G = MarkovGraph(Q, π)
    refs = [[2.0, 0.5, 0.5], [0.5, 2.0, 0.5], [0.5, 0.5, 2.0]]
    λ_true = [0.5, 0.3, 0.2]

    ν, _, _ = barycenter_socp(G, refs, λ_true; N=10)
    M = hcat(refs...)

    λ_socp = vec(analyze_socp(G, ν, refs; N=10))
    λ_mom  = vec(analyze_socp(G, ν, refs; N=10, convention=:momentum))
    λ_cp = vec(analysis(ν, M, Q; N=50, tol=1e-10))

    @test λ_socp ≈ λ_true atol=1e-3   # potential convention: exact stationarity, solver tol
    @test λ_mom ≈ λ_true atol=1e-2    # momentum convention: O(h) proxy
    @test λ_cp ≈ λ_true atol=1e-2
    @test λ_socp ≈ λ_cp atol=1e-2

    # The point of the potential convention: recovery does
    # not depend on N being fine. Re-synthesize and analyze at N=2, where the momentum
    # proxy is worst, and require the true λ to already be the QP's minimizer
    # (residual ratio λᵀAλ / λ̂ᵀAλ̂ ≈ 1).
    ν2, _, _ = barycenter_socp(G, refs, λ_true; N=2)
    λ̂2, A2 = analyze_socp(G, ν2, refs; N=2, return_system=true)
    λ̂2 = vec(λ̂2)
    @test λ̂2 ≈ λ_true atol=1e-3
    @test λ_true' * A2 * λ_true ≤ 1e-6 * maximum(diag(A2))   # true λ is (numerically) a zero of the Gram form
    @test λ̂2' * A2 * λ̂2 ≤ 1e-6 * maximum(diag(A2))
end

@testset "Hamiltonian shooting: conservation laws" begin
    # The three invariants that gate the Hamiltonian flow: mass
    # conservation, H conservation, and 2H == the squared discrete transport distance
    # between the flow's own endpoints (cross-checked against the independently
    # validated geodesic_socp) - i.e. the Hamiltonian flow really does trace a genuine
    # geodesic, not just some curve that happens to conserve H by construction.
    graphs = [triangle_markov_chain(), weighted_hypercube_markov_chain()]

    rng = MersenneTwister(1)
    for (Q, π) in graphs
        G = MarkovGraph(Q, π)
        for _ in 1:3
            ρ0 = (rand(rng, G.n) .+ 0.5); ρ0 ./= dot(ρ0, π)
            # Small scale: an arbitrary large random φ0 can drive some node's density
            # through zero within [0,1] (a real feature of this geometry's boundary
            # behavior, not a bug - see log_map_mollified) even
            # though ρ0 itself is safely interior. A genuine log_map-derived φ0 would
            # be commensurately small for a nearby target; this mimics that regime
            # without yet having exp_map/log_map built.
            φ0 = 0.1 .* randn(rng, G.n)
            φ0 .-= dot(φ0, π) .* ones(G.n)  # gauge: ⟨φ,1⟩_π = 0

            H0 = hamiltonian(G, ρ0, φ0)
            ρ_path, φ_path = integrate_hamiltonian(G, ρ0, φ0; nsteps=200, T=1.0)

            @test dot(ρ_path[:, end], π) ≈ 1.0 atol=1e-10  # mass conservation

            Hs = [hamiltonian(G, ρ_path[:, i], φ_path[:, i]) for i in 1:size(ρ_path, 2)]
            @test maximum(abs.(Hs .- H0)) < 1e-4  # H conservation (RK4 truncation error)

            # 2H vs geodesic_socp.W2 between the flow's own endpoints. Unlike the
            # larger-magnitude geodesic_socp gates above, φ0's small scale (see above) makes
            # 2H itself small (~0.01-0.04), so the residual here is dominated by
            # Clarabel's own solver-tolerance noise floor rather than a shrinking O(h)
            # truncation error - that floor doesn't shrink with N, so we check absolute
            # magnitude rather than requiring monotonic improvement across N.
            ρ_end = ρ_path[:, end]
            for N in (20, 100)
                W2 = geodesic_socp(G, ρ0, ρ_end; N=N).W2
                @test abs(2 * H0 - W2) < 1e-4
            end
        end
    end
end

@testset "Hamiltonian shooting vs two-node closed form" begin
    # Stronger, fully independent check than the geodesic_socp cross-check above: the
    # same closed-form two-node quadrature used to gate geodesic_socp above
    # (rho(r) = [1-r,1+r], W(rho(s),rho(t)) = (1/sqrt(2)) int_s^t (1-r^2)^(-1/4) dr).
    # On this graph the gauge-fixed potential reduces to a scalar φ = (c0,-c0), so we
    # can pick c0 directly (no log_map/shooting needed yet) and check that sqrt(2H0)
    # matches the quadrature distance between rho0 and wherever the flow actually
    # lands - no SOCP time-discretization error to hide behind here, unlike the
    # (already tight) check above.
    G = MarkovGraph([0.0 1.0; 1.0 0.0], [0.5, 0.5])
    W_ref(a, b) = quadgk(r -> (1 - r^2)^(-1/4), a, b)[1] / sqrt(2)

    r0 = -0.6
    ρ0 = [1 - r0, 1 + r0]

    for (c0, atol) in ((0.1, 1e-9), (0.3, 1e-8))
        φ0 = [c0, -c0]
        ρ_path, _ = integrate_hamiltonian(G, ρ0, φ0; nsteps=400, T=1.0)
        r_end = 1 - ρ_path[1, end]

        W_flow = sqrt(2 * hamiltonian(G, ρ0, φ0))
        W_quad = abs(W_ref(r0, r_end))
        @test W_flow ≈ W_quad atol=atol
    end

    # c0 large enough to drive r past the boundary -1 within [0,1]: the positivity
    # floor guard should catch this as an error, not silently return garbage or crash
    # with an uncaught DomainError.
    @test_throws PositivityFloorError integrate_hamiltonian(G, ρ0, [0.6, -0.6]; nsteps=400, T=1.0)
end

@testset "exp_map / weighted Laplacian" begin
    rng = MersenneTwister(2)
    Q, π = weighted_hypercube_markov_chain()
    G = MarkovGraph(Q, π)
    ν = rand(rng, G.n) .+ 0.5; ν ./= dot(ν, π)
    φ = 0.1 .* randn(rng, G.n); φ .-= dot(φ, π)

    @testset "π∘ρ̇ == L_θ(ν) φ" begin
        ρ̇, _ = hamiltonian_flow(G, ν, φ)
        @test weighted_laplacian(G, ν) * φ ≈ π .* ρ̇ rtol=1e-12
        @test weighted_laplacian(G, ν) * ones(G.n) ≈ zeros(G.n) atol=1e-14
    end

    @testset "momentum_to_potential inverts m = θ(ν)∘∇φ" begin
        m = metric_tensor(G, ν) .* graph_gradient(G, φ)
        φ_rec = momentum_to_potential(G, ν, m)
        @test φ_rec ≈ φ rtol=1e-10
        @test abs(dot(φ_rec, π)) < 1e-12   # gauge
    end

    @testset "exp_map agrees with integrate_hamiltonian for both tangent kinds" begin
        ρ_path, _ = integrate_hamiltonian(G, ν, φ; nsteps=100, T=1.0)
        m = metric_tensor(G, ν) .* graph_gradient(G, φ)
        @test exp_map(G, ν, φ; nsteps=100) ≈ ρ_path[:, end] rtol=1e-12
        @test exp_map(G, ν, φ .+ 3.0; nsteps=100) ≈ ρ_path[:, end] rtol=1e-12   # gauge-invariant
        @test exp_map(G, ν, m; nsteps=100) ≈ ρ_path[:, end] rtol=1e-8
        @test exp_map(G, ν, φ; nsteps=50, t=0.5) ≈ ρ_path[:, 51] rtol=1e-12   # same h as the 100-step path
        @test dot(exp_map(G, ν, φ; nsteps=100), π) ≈ 1.0 atol=1e-10
    end

    @testset "kind inference refuses the ambiguous n == |E| case" begin
        Q3, π3 = triangle_markov_chain()
        G3 = MarkovGraph(Q3, π3)
        ν3 = [1.2, 0.9, 0.9]; ν3 ./= dot(ν3, π3)
        φ3 = [0.05, -0.02, -0.03]
        @test_throws ArgumentError exp_map(G3, ν3, φ3)
        @test exp_map(G3, ν3, φ3; kind=:potential) ≈ integrate_hamiltonian(G3, ν3, φ3 .- dot(φ3, π3); nsteps=150)[1][:, end]
        @test_throws ArgumentError exp_map(G, ν, ones(5))
    end
end

@testset "log_map by shooting" begin
    rng = MersenneTwister(4)

    @testset "round trip, W2 and m0 vs geodesic_socp, Newton counts" begin
        for (Q, π) in (weighted_hypercube_markov_chain(), grid_markov_chain(5))
            G = MarkovGraph(Q, π)
            for _ in 1:3
                ν = rand(rng, G.n) .+ 0.5; ν ./= dot(ν, π)
                μ = rand(rng, G.n) .+ 0.5; μ ./= dot(μ, π)
                r = log_map(G, ν, μ)
                @test r.iters ≤ 8                                  # spec: 3-8 cold
                @test norm((exp_map(G, ν, r.φ0) .- μ) .* sqrt.(π)) < 1e-6   # spec (i)
                @test norm((exp_map(G, ν, r.m0) .- μ) .* sqrt.(π)) < 1e-6   # via the momentum too
                @test abs(dot(r.φ0, π)) < 1e-12                    # gauge
                @test log_map(G, ν, μ; φ0_init=r.φ0).iters == 0    # warm start from the solution
                # warm start toward a nearby target: no more iterations than a cold start
                μ2 = 0.98 .* μ .+ 0.02 .* ones(G.n)
                @test log_map(G, ν, μ2; φ0_init=r.φ0).iters ≤ log_map(G, ν, μ2).iters

                # m0 and W2 vs geodesic_socp, O(h) in the SOCP's h=1/N
                prev = Inf
                for N in (10, 40, 160)
                    sol = geodesic_socp(G, ν, μ; N=N)
                    m_err = norm(r.m0 .- sol.m0) / norm(sol.m0)
                    @test m_err < 5.0 / N
                    @test m_err < prev + 1e-6
                    @test abs(r.W2 - sol.W2) < 1.0 / N
                    prev = m_err
                end
                # The SOCP's endpoint potential is the gradient of W2, and the flow's φ0
                # is the Hamiltonian velocity potential: φ_socp ≈ -2 φ0 (continuum limit).
                # This (and the signed m0 comparison above) is what pins the flow's global
                # sign: the conservation-law tests pass equally for the time-reversed
                # flow, and the two-node closed-form checks take absolute values.
                sol = geodesic_socp(G, ν, μ; N=160)
                @test graph_gradient(G, sol.φ0) ≈ -2 .* graph_gradient(G, r.φ0) rtol=0.05
            end
        end
    end

    @testset "two-node closed form" begin
        G = MarkovGraph([0.0 1.0; 1.0 0.0], [0.5, 0.5])
        W_ref(a, b) = quadgk(r -> (1 - r^2)^(-1/4), a, b)[1] / sqrt(2)
        s, t = -0.6, 0.7
        r = log_map(G, [1 - s, 1 + s], [1 - t, 1 + t]; nsteps=400)
        @test sqrt(r.W2) ≈ W_ref(s, t) atol=1e-8
    end

    @testset "far-apart concentrated endpoints (damped initialization)" begin
        # The linearized initial guess overshoots through the positivity floor here; the
        # damped initialization must recover and Newton must still converge.
        Q, π = grid_markov_chain(5)
        G = MarkovGraph(Q, π)
        A = Q .> 0
        conc(c) = (m = ones(G.n); m[c] *= 10; for j in 1:G.n; A[c, j] && (m[j] *= 10); end; m ./ dot(m, π))
        ν, μ = conc(1), conc(25)
        # Premise: the undamped linearized guess really does hit the floor on this input
        # (otherwise this testset would silently stop exercising the damping loop).
        φ0_lin = solve_weighted_laplacian(G, ν, π .* (μ .- ν))
        @test_throws PositivityFloorError integrate_hamiltonian(G, ν, φ0_lin; nsteps=150)
        r = log_map(G, ν, μ)
        @test r.residual < 1e-9
        @test norm((exp_map(G, ν, r.φ0) .- μ) .* sqrt.(π)) < 1e-6
        @test abs(r.W2 - geodesic_socp(G, ν, μ; N=100).W2) < 2e-2
    end

    @testset "guards" begin
        Q, π = triangle_markov_chain()
        G = MarkovGraph(Q, π)
        @test_throws AssertionError log_map(G, [1.0, 1.0, 1.0], [0.0, 1.5, 1.5])   # zero entry
        @test_throws AssertionError log_map(G, [1.0, 1.0, 1.0], [2.0, 1.0, 1.0])   # not a density
    end
end

@testset "potential_gram_qp: Gram matrix is the Riemannian inner product at the target" begin
    # Every recovery test uses an exactly stationary synthesized target, where the true λ
    # minimizes the QP under *any* positive edge weighting - so none of them can tell
    # κ∘θ from θ from 1. Pin the weighting directly against the dense definition
    #   A_ij = ½ Σ_{x,y} θ(ν_x,ν_y) ∇φ_i(x,y) ∇φ_j(x,y) Q(x,y) π(x).
    Q, π = triangle_markov_chain()
    G = MarkovGraph(Q, π)
    ν = [1.3, 0.8, 0.9]; ν ./= dot(ν, π)
    φs = [[0.3, -0.1, -0.2], [0.0, 0.5, -0.5]]
    _, A = GraphTransportation.potential_gram_qp(G, ν, φs; return_system=true)
    θ = metric_tensor(ν)
    A_ref = [0.5 * sum(θ[x, y] * (φi[x] - φi[y]) * (φj[x] - φj[y]) * Q[x, y] * π[x]
                       for x in 1:3, y in 1:3)
             for φi in φs, φj in φs]
    @test A ≈ A_ref rtol=1e-12
    @test A ≈ A' rtol=1e-12
end

@testset "analyze_shooting (:shooting analysis backend)" begin
    Q, π = weighted_hypercube_markov_chain()
    G = MarkovGraph(Q, π)
    rng = MersenneTwister(6)
    refs = [(v = rand(rng, G.n) .+ 0.3; v ./= dot(v, π)) for _ in 1:3]
    λ_true = [0.5, 0.3, 0.2]

    # Synthesized by the SOCP at fine N: the shooting backend checks stationarity in a
    # different discretization, so expect O(1/N) agreement, not solver tolerance.
    # Check the rate, not just a single small error. The recovered λ̂ itself is a poor
    # rate probe: it comes out of the SCS simplex QP, whose default tolerance leaves a
    # platform-dependent floor of ~1e-5 to ~5e-4 on |λ̂-λ| (Julia 1.10 on CI sits at the
    # top of that range), so |λ̂-λ| stops shrinking with N almost immediately. The
    # quantity that is genuinely O(h) and involves no QP is the Gram-form residual of
    # the *true* λ, λᵀAλ / max(diag A): measured 8e-7 to 1.3e-6 at N=2 and 4e-9 to 7e-9
    # at N=10 on Julia 1.10/1.12 (a factor of 100-300). Require a factor of 10.
    resid = Float64[]
    local ν, λ̂
    for N in (2, 10)
        ν, _, _ = barycenter_socp(G, refs, λ_true; N=N)
        λ̂_, A = analyze_shooting(G, ν, refs; return_system=true)
        λ̂ = vec(λ̂_)
        push!(resid, (λ_true' * A * λ_true) / maximum(diag(A)))
    end
    @test resid[2] < resid[1] / 10
    @test λ̂ ≈ λ_true atol=1e-2        # loose: QP floor, see above
    @test sum(λ̂) ≈ 1.0 atol=1e-6

    # Same point, same reference potentials from the two backends: at fine N the SOCP's
    # endpoint duals and the flow's φ0 give the same Gram matrix up to the factor (-2)²
    # and O(h), so the two backends must agree closely with each other.
    λ_socp = vec(analyze_socp(G, ν, refs; N=80))
    @test λ̂ ≈ λ_socp atol=1e-2

    # Warm starts are accepted and don't change the answer.
    inits = [log_map(G, ν, r).φ0 for r in refs]
    @test vec(analyze_shooting(G, ν, refs; φ0_inits=inits)) ≈ λ̂ atol=1e-8
end

@testset "log_map_mollified (boundary fallback)" begin
    # Target supported on two columns of a 5x5 grid (zero elsewhere): shooting cannot
    # run directly, so mollify and extrapolate. geodesic_socp handles the boundary case
    # natively and is the reference. Both the extrapolated and the raw smallest-ε
    # distance should be within a percent of it (the fallback is approximate by design).
    Q, π = grid_markov_chain(5)
    G = MarkovGraph(Q, π)
    rng = MersenneTwister(7)
    ν = rand(rng, G.n) .+ 0.5; ν ./= dot(ν, π)
    tgt = zeros(G.n)
    for i in 1:G.n
        mod(i - 1, 5) < 2 && (tgt[i] = 1.0 + rand(rng))
    end
    tgt ./= dot(tgt, π)
    @test_throws AssertionError log_map(G, ν, tgt)

    W_ref = sqrt(geodesic_socp(G, ν, tgt; N=400).W2)
    r = log_map_mollified(G, ν, tgt)
    @test r.approximate
    @test length(r.Ws) ≥ 2                    # a stiff level may be skipped (seen on Julia 1.10)
    @test issorted(r.Ws)                       # W increases as ε → 0 (less smoothing)
    @test abs(r.W - W_ref) / W_ref < 1e-2
    @test abs(r.Ws[end] - W_ref) / W_ref < 5e-3
    @test r.W2 ≈ r.W^2
end

@testset "chambolle_pock (fixed-step Algorithm 1) converges to the SOCP-validated value" begin
    # Chambolle-Pock's accelerated Algorithm 2 schedule requires G or F* to be strongly
    # convex; every term here (the K-cone / continuity-equation / J_Eq indicators, the
    # homogeneous-degree-1 edge action) is not, and applying it anyway converged to a
    # biased value on every graph with more than 2 nodes. The routine is now fixed-step
    # only; this test ties it to the independently validated geodesic_socp so a
    # regression of that kind cannot come back unnoticed.
    Q, π = triangle_markov_chain()
    G = MarkovGraph(Q, π)
    rng = MersenneTwister(2024)
    μ = (rand(rng, 3) .+ 0.1); μ ./= dot(μ, π)
    ν = (rand(rng, 3) .+ 0.1); ν ./= dot(ν, π)

    prev_err = Inf
    for N in (10, 20, 50, 100)
        W2_socp = geodesic_socp(G, μ, ν; N=N).W2
        W2_cp = action(discrete_transport(Q, μ, ν; N=N, tol=1e-12, maxiters=2^20))
        err = abs(W2_cp - W2_socp) / W2_socp
        @test err < 2.0 / N          # O(h), same pattern as the two-node gate
        @test err < prev_err + 1e-9  # should not grow with N
        prev_err = err
    end
end

@testset "accept_descent_step: noise-scale negatives are not overshoots" begin
    # The WGD step-halving loop used to reject any ν_next with a strictly negative
    # entry. Solver noise in the Chambolle-Pock momenta can put an entry at ~-1e-11
    # when the true value is 0, and halving cannot fix noise of that size, so the loop
    # would exhaust its retries and error spuriously on near-boundary barycenters.
    π = [0.25, 0.25, 0.5]
    ν = [1.0, 0.0, 1.5]              # ⟨ν, π⟩ = 1, one entry exactly at the boundary
    h = 0.1

    # Mass-preserving direction (⟨d, π⟩ = 0) whose step ν - h·d puts node 2 at -1e-11:
    # noise scale, not an overshoot.
    d_noise = [-4e-10, 1e-10, 1.5e-10]
    @test abs(dot(d_noise, π)) < 1e-20
    ν_next = GraphTransportation.accept_descent_step(ν, d_noise, h, π)
    @test minimum(ν_next) == 0.0                    # clamped, not rejected
    @test ν_next ≈ ν .- h .* d_noise atol=1e-10     # full step accepted, no halving
    @test abs(dot(ν_next, π) - 1) < 1e-8

    # Genuine overshoot from an interior point: full step drives node 2 to -0.4; three
    # halvings (h/8) bring it to 0.1 - 0.0625 = 0.0375 ≥ 0.
    ν_int = [1.0, 0.1, 1.45]                        # ⟨ν_int, π⟩ = 1
    d_big = [10.0, 5.0, -7.5]                       # ⟨d_big, π⟩ = 0
    @test abs(dot(d_big, π)) < 1e-12
    ν_next = GraphTransportation.accept_descent_step(ν_int, d_big, h, π)
    @test ν_next ≈ ν_int .- (h / 8) .* d_big
    @test abs(dot(ν_next, π) - 1) < 1e-8

    # From a node at exactly zero, an outward direction cannot be rescued by halving
    # (every step size goes negative): still an error, as before.
    @test_throws ErrorException GraphTransportation.accept_descent_step(ν, d_big, h, π)
    # Mass violation that halving cannot repair (⟨d, π⟩ ≠ 0) still errors.
    @test_throws ErrorException GraphTransportation.accept_descent_step(ν, [1.0, 1.0, 1.0], h, π)
end

@testset "unified API: geodesic / transport_cost / barycenter / analysis with method=" begin
    Q, π = grid_markov_chain(4)
    G = MarkovGraph(Q, π)
    rng = MersenneTwister(9)
    ρA = rand(rng, G.n) .+ 0.5; ρA ./= dot(ρA, π)
    ρB = rand(rng, G.n) .+ 0.5; ρB ./= dot(ρB, π)

    @testset "geodesic: all methods return a GeodesicSolution with consistent W2" begin
        g_socp = geodesic(G, ρA, ρB; N=40)
        g_sh   = geodesic(G, ρA, ρB; method=:shooting)
        g_cp   = geodesic(G, ρA, ρB; method=:chambolle_pock, N=40, tol=1e-11, maxiters=2^18)
        for g in (g_socp, g_sh, g_cp)
            @test g isa GeodesicSolution
            @test size(g.ρ, 1) == G.n && size(g.m, 1) == length(G.E)
            @test g.ρ[:, 1] ≈ ρA atol=1e-6
            @test g.ρ[:, end] ≈ ρB atol=1e-6
            @test g.m0 == g.m[:, 1]
        end
        @test g_socp.status == OPTIMAL && g_sh.status == :converged && g_cp.status == :converged
        @test abs(g_socp.W2 - g_sh.W2) < 2e-2 * g_sh.W2      # O(1/N) at N=40
        @test abs(g_cp.W2 - g_sh.W2) < 5e-2 * g_sh.W2
        @test norm(g_socp.m0 .- g_sh.m0) / norm(g_sh.m0) < 0.1
        # Chambolle-Pock's momentum agrees with the others away from the endpoints, but its
        # first-interval momentum is a boundary-cell artifact of the Galerkin scheme (~50%
        # off, not shrinking with N); see the `geodesic` docstring. Compare mid-path.
        mid = size(g_socp.m, 2) ÷ 2
        @test norm(g_cp.m[:, mid] .- g_socp.m[:, mid]) / norm(g_socp.m[:, mid]) < 0.05
        @test_broken norm(g_cp.m0 .- g_sh.m0) / norm(g_sh.m0) < 0.15
        @test all(isnan, g_cp.φ0) && all(isnan, g_cp.φ1)
        # shooting's endpoint potentials use the W2-gradient convention of the SOCP duals
        g_fine = geodesic(G, ρA, ρB; N=160)
        @test graph_gradient(G, g_sh.φ0) ≈ graph_gradient(G, g_fine.φ0) rtol=0.05
        @test graph_gradient(G, g_sh.φ1) ≈ graph_gradient(G, g_fine.φ1) rtol=0.05
        @test transport_cost(G, ρA, ρB; N=40) ≈ sqrt(g_socp.W2)
        @test transport_cost(G, ρA, ρB; method=:shooting) ≈ sqrt(g_sh.W2)
        @test_throws ArgumentError geodesic(G, ρA, ρB; method=:entropic)     # unknown method
        @test_throws ArgumentError geodesic(G, ρA, ρB; method=:sinkhorn)      # cost/epsilon required
    end

    @testset "barycenter and analysis dispatch" begin
        refs = [ρA, ρB, (v = rand(rng, G.n) .+ 0.5; v ./= dot(v, π))]
        λ = [0.5, 0.3, 0.2]
        ν, J, info = barycenter(G, refs, λ; N=10)
        @test info.geodesics isa Vector{GeodesicSolution} && length(info.geodesics) == 3
        @test J ≈ sum(λ[i] * geodesic(G, refs[i], ν; N=10).W2 for i in 1:3) rtol=1e-4
        @test vec(analysis(G, ν, refs; N=10)) ≈ λ atol=1e-3
        @test vec(analysis(G, ν, refs; method=:shooting)) ≈ λ atol=5e-2
        @test vec(analysis(G, ν, refs; method=:chambolle_pock, N=10, tol=1e-10)) ≈ λ atol=5e-2
        λ̂, A = analysis(G, ν, refs; N=10, return_system=true)
        @test size(A) == (3, 3) && A ≈ A'

        ν_cp, J_cp, info_cp = barycenter(G, refs, λ; method=:chambolle_pock, h=0.1, maxiters=30,
                                          geodesic_steps=10, geodesic_tol=1e-8, verbose=false)
        @test haskey(info_cp, :variances) && abs(dot(ν_cp, π) - 1) < 1e-8 && minimum(ν_cp) ≥ 0
        # optimality certificate, with both objectives evaluated by the same geodesic solver
        J_at(ν) = sum(λ[i] * geodesic(G, refs[i], ν; N=10).W2 for i in 1:3)
        @test J_at(ν_cp) ≥ J_at(ν) - 1e-6
        @test_throws ArgumentError barycenter(G, refs, λ; method=:shooting)
    end
end

@testset "Sinkhorn barycentric coordinates (Bonneel, Peyré & Cuturi 2016)" begin
    # Small instance: 3x3 grid, shortest-path-squared cost normalized to [0,1].
    Q, π = grid_markov_chain(3); n = 9
    A = Q .> 0
    D = fill(Inf, n, n); for i in 1:n; D[i, i] = 0.0; end
    for i in 1:n, j in 1:n; A[i, j] && (D[i, j] = 1.0); end
    for k in 1:n, i in 1:n, j in 1:n; D[i, j] = min(D[i, j], D[i, k] + D[k, j]); end
    cost = (D .^ 2) ./ maximum(D .^ 2)
    rng = MersenneTwister(1)
    M = rand(rng, n, 3) .+ 0.2; M ./= sum(M, dims=1)
    q = rand(rng, n) .+ 0.2; q ./= sum(q)
    ε = 0.1
    λ = [0.5, 0.3, 0.2]
    SK = GraphTransportation

    @testset "barycenter is a probability vector; permutation-equivariant" begin
        p = sinkhorn_barycenter(λ, M, nothing, cost, ε; iters=64)
        @test sum(p) ≈ 1.0 atol=1e-10
        @test all(p .> 0)
        p2 = sinkhorn_barycenter(λ[[2, 1, 3]], M[:, [2, 1, 3]], nothing, cost, ε; iters=64)
        @test p2 ≈ p rtol=1e-12
    end

    # Algorithm 1's gradient w = ∇_λ E_L against central finite differences of the same
    # finite-L objective (the paper's own check, Fig. 5), at a small L where the reverse
    # loop's bounds matter (an off-by-one gave the wrong sign at L=2) and at convergence.
    @testset "∇_λ E_L matches finite differences" begin
        E(λv, L) = SK.sqeuc_loss(SK.sinkhorn_differentiate(λv, M, q, cost, ε, L)[1], q)
        h = 1e-6
        for L in (3, 6, 60)
            fd = [(E(λ .+ h .* (1:3 .== i), L) - E(λ .- h .* (1:3 .== i), L)) / (2h) for i in 1:3]
            _, w = SK.sinkhorn_differentiate(λ, M, q, cost, ε, L)
            @test w ≈ fd rtol=1e-6
        end
    end

    # The regression variable is α with λ = softmax(α); the gradient handed to L-BFGS
    # must include the softmax Jacobian (it did not, and was ~28x off).
    @testset "∇_α E matches finite differences through the softmax" begin
        α = [0.2, -0.1, 0.3]
        Eα(a) = SK.barycentric_loss(a, M, q, cost, ε; iters=40)
        h = 1e-6
        fd = [(Eα(α .+ h .* (1:3 .== i)) - Eα(α .- h .* (1:3 .== i))) / (2h) for i in 1:3]
        @test SK.loss_gradient(α, M, cost, q, ε; iters=40) ≈ fd rtol=1e-6
        @test abs(sum(SK.loss_gradient(α, M, cost, q, ε; iters=40))) < 1e-12   # tangent to the simplex
    end

    @testset "simplex_regression recovers the synthesis weights" begin
        target = sinkhorn_barycenter(λ, M, nothing, cost, ε; iters=256)
        λ̂ = simplex_regression(M, target, cost, ε; iters=256)
        @test sum(λ̂) ≈ 1.0 atol=1e-10
        @test λ̂ ≈ λ atol=1e-3
    end
end

@testset "barycenter(method=:sinkhorn) and ground_cost" begin
    G = MarkovGraph(grid_markov_chain(3)...)
    C = ground_cost(G, :shortest_path)
    @test C ≈ C' && all(iszero, diag(C))
    @test C[1, 9] == 1.0 && C[1, 2] == 1 / 16          # corner-to-corner (4 hops)², adjacent 1²/4²
    @test graph_diameter(G) == 4
    Cd = ground_cost(G, :diffusion)
    @test Cd ≈ Cd' && all(iszero, diag(Cd)) && maximum(Cd) == 1.0 && all(isfinite, Cd)
    @test minimum(Cd[i, j] for i in 1:9, j in 1:9 if i != j) > 0
    @test_throws ArgumentError ground_cost(G, :euclidean)

    refs = [[3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 3.0]]
    refs = [r ./ dot(r, G.π) for r in refs]
    λ = [0.5, 0.5]; ε = 0.05
    ν, J, info = barycenter(G, refs, λ; method=:sinkhorn, cost=C, epsilon=ε)
    # equivalence pin: the wrapper is exactly the old call under the density/probability conversion
    @test ν .* G.π ≈ sinkhorn_barycenter(λ, reduce(hcat, (r .* G.π for r in refs)), nothing, C, ε)
    @test dot(ν, G.π) ≈ 1.0 atol=1e-10
    @test all(ν .≥ 0)
    @test J ≥ 0 && all(info.marginal_errors .< 1e-8)
    @test info.epsilon == ε && info.cost === C
    @test_throws ArgumentError barycenter(G, refs, λ; method=:sinkhorn, epsilon=ε)
    @test_throws ArgumentError barycenter(G, refs, λ; method=:sinkhorn, cost=C)

    @testset "geodesic(method=:sinkhorn): entropic displacement interpolation" begin
        g = geodesic(G, refs[1], refs[2]; method=:sinkhorn, cost=C, epsilon=ε, N=4)
        @test g isa GeodesicSolution && size(g.ρ) == (G.n, 5)
        @test all(abs(dot(g.ρ[:, k], G.π) - 1) < 1e-10 for k in 1:5)
        @test all(g.ρ .≥ 0) && all(isnan, g.m) && all(isnan, g.φ0)
        @test g.W2 ≥ 0 && g.W2 ≈ dot(C, GraphTransportation._sinkhorn_plan(GraphTransportation.regularize_cost(C, ε), refs[1] .* G.π, refs[2] .* G.π))
        # the t=0 / t=1 columns are the Sinkhorn barycenters with weights (1,0) / (0,1)
        @test g.ρ[:, 1] ≈ barycenter(G, refs, [1.0, 0.0]; method=:sinkhorn, cost=C, epsilon=ε)[1]
        @test g.ρ[:, 5] ≈ barycenter(G, refs, [0.0, 1.0]; method=:sinkhorn, cost=C, epsilon=ε)[1]
        # time reversal: swapping the endpoints reverses the path
        g_rev = geodesic(G, refs[2], refs[1]; method=:sinkhorn, cost=C, epsilon=ε, N=4)
        @test g_rev.ρ ≈ g.ρ[:, end:-1:1] rtol=1e-8
        @test transport_cost(G, refs[1], refs[2]; method=:sinkhorn, cost=C, epsilon=ε) ≈ sqrt(g.W2)
        @test_throws ArgumentError geodesic(G, refs[1], refs[2]; method=:sinkhorn, epsilon=ε)
    end

    @testset "analysis(method=:sinkhorn): recovers the synthesis weights" begin
        refs3 = [refs[1], refs[2], (v = [0.5, 0.5, 0.5, 0.5, 3.0, 0.5, 0.5, 0.5, 0.5]; v ./ dot(v, G.π))]
        λ3 = [0.5, 0.3, 0.2]
        ν3, _, _ = barycenter(G, refs3, λ3; method=:sinkhorn, cost=C, epsilon=ε)
        λ̂ = analysis(G, ν3, refs3; method=:sinkhorn, cost=C, epsilon=ε)
        @test sum(λ̂) ≈ 1.0 atol=1e-10
        @test λ̂ ≈ λ3 atol=1e-3
        @test_throws ArgumentError analysis(G, ν3, refs3; method=:sinkhorn, cost=C)
        @test_throws ArgumentError analysis(G, ν3, refs3; method=:sinkhorn, cost=C, epsilon=ε, return_system=true)
    end
end

@testset "Admissible means (types, derivatives, admissibility)" begin
    means = (GeometricMean(), ArithmeticMean(), HarmonicMean(), LogarithmicMean(), QuadLogMean(8))
    rng = MersenneTwister(12)
    pts = [(rand(rng) * 3 + 0.05, rand(rng) * 3 + 0.05) for _ in 1:50]

    @testset "$(θ)" for θ in means
        for (s, t) in pts
            @test θ(s, t) ≈ θ(t, s)                                   # symmetric
            @test θ(2.5s, 2.5t) ≈ 2.5 * θ(s, t)                        # 1-homogeneous
            @test θ(s, t) > 0
            @test θ(s, t) ≤ (s + t) / 2 + 1e-12                        # ≤ arithmetic
            # concavity (midpoint inequality) against a second random point
            (s2, t2) = pts[mod1(hash((s, t)) % 50 + 1, 50)]
            @test θ((s + s2) / 2, (t + t2) / 2) ≥ (θ(s, t) + θ(s2, t2)) / 2 - 1e-12
            # ∂₁θ against ForwardDiff
            @test partial_s(θ, s, t) ≈ ForwardDiff.derivative(u -> θ(u, t), s) rtol=1e-10
            @test partial_t(θ, s, t) ≈ ForwardDiff.derivative(u -> θ(s, u), t) rtol=1e-10
        end
        @test θ(1.7, 1.7) ≈ 1.7
        @test partial_s(θ, 1.7, 1.7) ≈ 0.5 atol=1e-12                # symmetric ⇒ ∂₁θ(s,s) = 1/2
    end

    @testset "ordering harmonic ≤ geometric ≤ logarithmic ≤ arithmetic" begin
        for (s, t) in pts
            @test HarmonicMean()(s, t) ≤ GeometricMean()(s, t) + 1e-12
            @test GeometricMean()(s, t) ≤ LogarithmicMean()(s, t) + 1e-12
            @test LogarithmicMean()(s, t) ≤ ArithmeticMean()(s, t) + 1e-12
        end
        @test GeometricMean()(2.0, 8.0) == 4.0 && HarmonicMean()(2.0, 8.0) == 3.2 && ArithmeticMean()(2.0, 8.0) == 5.0
        @test LogarithmicMean()(1.0, ℯ) ≈ ℯ - 1
    end

    @testset "logarithmic mean: series branch is continuous with the closed form" begin
        Λ = LogarithmicMean()
        exact(s, t) = (s - t) / (log(s) - log(t))
        for δ in (3e-3, 2e-3, 5e-4, 5e-5, 1e-6, -2e-3, -5e-4)   # straddle the 1e-3 switch
            @test Λ(1.0 + δ, 1.0) ≈ exact(1.0 + δ, 1.0) rtol=1e-13
            @test partial_s(Λ, 1.0 + δ, 1.0) ≈ ForwardDiff.derivative(u -> exact(u, 1.0), 1.0 + δ) rtol=1e-9
        end
        @test Λ(3.0, 3.0) == 3.0 && partial_s(Λ, 3.0, 3.0) == 0.5
    end

    @testset "QuadLogMean(K) converges to the logarithmic mean" begin
        Λ = LogarithmicMean()
        worst(K) = maximum(abs(QuadLogMean(K)(r, 1.0) - Λ(r, 1.0)) / Λ(r, 1.0) for r in (1.5, 3, 10, 30, 100, 1000))
        @test worst(4) < 1e-3 && worst(6) < 1e-6 && worst(8) < 1e-9 && worst(12) < 1e-13
        @test worst(4) > worst(6) > worst(8) > worst(12)
        q = QuadLogMean(8)
        @test sum(q.w) ≈ 1.0 && q.α ≈ 1 .- reverse(q.α)                # symmetric rule ⇒ symmetric mean
        @test_throws ArgumentError QuadLogMean(0)
    end

    @testset "metric_tensor accepts an AdmissibleMean" begin
        Q, π = triangle_markov_chain()
        G = MarkovGraph(Q, π)
        ρ = [1.5, 0.8, 0.7]
        for θ in means
            edge = metric_tensor(G, ρ, θ)
            dense = metric_tensor(ρ, θ)
            @test edge ≈ [dense[x, y] for (x, y) in G.E]
        end
        @test metric_tensor(G, ρ, GeometricMean()) ≈ metric_tensor(G, ρ)     # default is geometric
    end
end

@testset "SOCP with each admissible mean" begin
    # Generalized two-node closed form: with ρ(r) = [1−r, 1+r] and π = [½,½],
    #   W(ρ(s), ρ(t)) = (1/√2) ∫_s^t θ(1−r, 1+r)^(−1/2) dr
    # (the geometric case is the (1−r²)^(−1/4) integrand used to gate Module 1). This
    # pins each mean's conic form, including its constants, independently of the SOCP.
    G2 = MarkovGraph([0.0 1.0; 1.0 0.0], [0.5, 0.5])
    s, t = -0.6, 0.7
    ρA = [1 - s, 1 + s]; ρB = [1 - t, 1 + t]
    W_ref(θ) = quadgk(r -> θ(1 - r, 1 + r)^(-1/2), s, t)[1] / sqrt(2)

    @testset "two-node closed form: $(θ)" for θ in (GeometricMean(), ArithmeticMean(), HarmonicMean(), QuadLogMean(8))
        C = W_ref(θ)
        prev = Inf
        for N in (10, 40, 160)
            sol = geodesic_socp(G2, ρA, ρB; N=N, mean=θ)
            @test sol.status == OPTIMAL
            err = abs(sqrt(sol.W2) - C)
            @test err < 2.0 / N
            @test err < prev + 1e-6
            prev = err
        end
    end
    @test ArithmeticMean()(1 - 0.3, 1 + 0.3) == 1.0   # constant mobility on two nodes ⇒ W = (t−s)/√2
    @test sqrt(geodesic_socp(G2, ρA, ρB; N=5, mean=ArithmeticMean()).W2) ≈ (t - s) / sqrt(2) atol=1e-6
    @test_throws ArgumentError geodesic_socp(G2, ρA, ρB; N=5, mean=LogarithmicMean())

    Q, π = triangle_markov_chain()
    G = MarkovGraph(Q, π)
    a = [2.0, 0.5, 0.5]; b = [0.5, 0.5, 2.0]
    @testset "distance ordering harmonic ≥ geometric ≥ logarithmic ≥ arithmetic" begin
        W2 = Dict(name => geodesic_socp(G, a, b; N=20, mean=θ).W2
                  for (name, θ) in (("H", HarmonicMean()), ("G", GeometricMean()), ("L", QuadLogMean(8)), ("A", ArithmeticMean())))
        @test W2["H"] > W2["G"] > W2["L"] > W2["A"]
        # K=8 vs K=12 quadrature agree to solver tolerance
        @test geodesic_socp(G, a, b; N=20, mean=QuadLogMean(12)).W2 ≈ W2["L"] rtol=1e-6
    end

    @testset "barycenter + analysis round trip per mean" begin
        refs = [[2.0, 0.5, 0.5], [0.5, 2.0, 0.5], [0.5, 0.5, 2.0]]
        λ = [0.5, 0.3, 0.2]
        for θ in (ArithmeticMean(), HarmonicMean(), QuadLogMean(8))
            ν, J, geos = barycenter_socp(G, refs, λ; N=6, mean=θ)
            @test abs(dot(ν, π) - 1) < 1e-8 && minimum(ν) ≥ -1e-8
            @test J ≈ sum(λ[i] * geodesic_socp(G, refs[i], ν; N=6, mean=θ).W2 for i in 1:3) rtol=1e-4
            λ̂ = vec(analyze_socp(G, ν, refs; N=6, mean=θ))
            @test λ̂ ≈ λ atol=2e-3
            # Σλᵢφ1ᵢ = const (KKT stationarity) holds for any mean
            stationarity = sum(λ[i] .* geos[i].φ1 for i in 1:3)
            @test maximum(abs, graph_gradient(G, stationarity)) < 1e-5 * maximum(abs, graph_gradient(G, geos[1].φ1))
        end
        # analysing with the wrong mean is a convention mismatch, not exact
        ν_h, _, _ = barycenter_socp(G, refs, λ; N=6, mean=HarmonicMean())
        @test norm(vec(analyze_socp(G, ν_h, refs; N=6, mean=ArithmeticMean())) .- λ) > 1e-3
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
