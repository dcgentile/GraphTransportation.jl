# Reviewing PR #23 (Hamiltonian shooting): notes from a guided walkthrough

Notes from reviewing https://github.com/dcgentile/GraphTransportation.jl/pull/23
step by step, 2026-09-16. Two purposes: (1) a record of what was actually
verified and what remains to be posted as the review, (2) a reference for the
math behind `src/shooting/`, written up in response to the questions asked
along the way. Questions are quoted as asked.

---

## 1. How to review (the framework used)

A review answers four questions, in order:

1. **Does it claim the right thing?** PR description vs. spec, before reading code.
2. **How would I know if it were wrong?** Decide what evidence would convince
   you, then check whether the PR supplies it. For numerical code: which checks
   are against an *independent* reference, and which are only self-consistency?
3. **Where would a bug hide?** Sign conventions, time indexing, gauge freedom,
   boundary cases, error paths that could silently return garbage. Look there.
4. **Would I want to maintain this?** Naming, honest docstrings, dead code,
   knobs nobody will set.

Habits: read one commit at a time (the commits were built as reviewable units);
keep a running notes file; never suppress a naive question ("why is there a
factor of 2 here" is the most productive comment in numerical code).

Label every comment as one of:

- **Finding**: the code produces a wrong result or fails a case it should handle.
- **Coverage note**: a test would not catch some plausible regression.
- **Description gap**: the PR text omits or overstates something.
- **Taste**: would have written it differently.

Findings first, then coverage, then taste. Mixing them up either blocks merges
over style or waves through bugs buried among nitpicks.

Calibration: 775 lines (≈400 of implementation) is large-ish for one PR; the
comfortable range is 100–300. A practiced reviewer spends 45–90 min on this
one. Speed comes from knowing the codebase's conventions (done once), knowing
which lines deserve scrutiny (~20% of lines get ~80% of attention), and reading
tests first for the risky parts. Core numerical code that everything depends on
gets this depth; a benchmark script or a rename gets five minutes.

---

## 2. Step 1: description vs. spec

Route: (1) description vs. spec; (2) `Hamiltonian.jl`; (3) `ExpLog.jl` helpers
+ `exp_map`; (4) `log_map`; (5) mollification + `analyze_shooting`; (6) tests
as an adversary; (7) benchmark + wiring, then write the review.

Answers given and what a reviewer would add:

- **Explicit skip**: multiple shooting (spec 3.3). Acceptable for this PR as
  long as single shooting fails *loudly* on the stiff cases (it does; see §5).
- **Quiet skip not caught**: spec 3.4 asks for a warm-start *cache* keyed by
  (base, target); the PR provides only a `φ0_init` argument. → description gap.
- **Which number matters most**: the round-trip error looks most impressive but
  is the *least* informative: `exp∘log ≈ id` tests that Newton solved the
  equation it was given, and would pass even with the wrong Hamiltonian. The
  independent evidence is the two-node closed form and the SOCP's `W2`/`m0`.
- The `analyze_shooting` vs `barycenter_socp` agreement of 1e-2 at N=80 is
  consistent with the claimed O(h) (1/80 ≈ 0.0125), but the test checks one N
  only; the rate claim is asserted, not tested. → **finding #1**.
- **Duplication check**: none of #22's three commit titles appear in #23's six;
  "Files changed" contains no `src/socp/Geodesic.jl` or `Barycenter.jl`. Clean.

---

## 3. Step 2: `src/shooting/Hamiltonian.jl`

> "help me understand this Hamiltonian formulation, and the subscript
> notation. are those different functions ρ and φ, or are those the values of
> functions on nodes labeled 1, 2?"

Values of functions on nodes. ρ and φ are each one scalar per vertex
(`Vector{Float64}` of length `G.n`; `ρ[x]` is ρ(x)). ρ₁ is shorthand for ρ(1).
The state (ρ, φ) lives in Rⁿ × Rⁿ.

- **φ** is a potential; its edge gradient (∇φ)(x,y) = φ(x) − φ(y) is a velocity
  field. Every tangent vector at ρ is a gradient of some potential, modulo
  constants (the *gauge freedom*: adding a constant to φ changes nothing). In
  Hamiltonian language φ is the momentum conjugate to ρ.
- **H(ρ, φ) = ½ Σ_e κ_e θ(ρ_x, ρ_y) (∇φ)_e²** is half the squared Riemannian
  speed: kinetic energy. θ is the mobility (geometric mean), κ_e = Q(x,y)π(x)
  the edge weight, the ½ (vs. the ordered-pair form) corrects for double
  counting. `hamiltonian` computes exactly this.
- Geodesics of a Riemannian metric are the trajectories of the Hamiltonian flow
  of its kinetic energy. That is the whole idea of shooting.
- Hamilton's equations with the π-weighted pairing:
  ρ̇(x) = (1/π(x)) ∂H/∂φ(x), φ̇(x) = −(1/π(x)) ∂H/∂ρ(x).

**Two-node derivation** (κ = Q(1,2)π(1)): H = ½κθ(φ(1)−φ(2))²;
∂H/∂φ(1) = κθ(φ(1)−φ(2)); ρ̇(1) = Q(1,2) θ (φ(1) − φ(2)). Mass flows *into*
node 1 when φ(1) > φ(2).

> "the paper computation I follow fine, I'm still a bit confused about what
> I'm looking for in the code base"

Pick one claim the code makes and trace the smallest set of definitions needed
to confirm it, substituting concrete values (2 nodes, 1 edge). Here the claim is
"`ρ̇ = -graph_divergence(G, θ .* ∇φ)` (Hamiltonian.jl:52) equals the spec's
formula". Lines involved: `MarkovGraph.jl:78` (∇φ[e] = φ[x] − φ[y]),
`MarkovGraph.jl:62-63` (D[x,e] = −Q[x,y], D[y,e] = +Q[y,x]),
`MarkovGraph.jl:94` (div m = D m). Substituting for e = (1,2):
m[e] = θ(φ(1)−φ(2)); (div m)[1] = −Q(1,2)·m[e]; ρ̇(1) = +Q(1,2)θ(φ(1)−φ(2)). ✓
Node 2 gives Q(2,1)θ(φ(2)−φ(1)) ✓, and π(1)ρ̇(1) + π(2)ρ̇(2) = 0 by
reversibility, so mass is conserved by the structure of the equation.
Record: "Checked ρ̇ sign by hand on the 2-node graph against D at
MarkovGraph.jl:62. Correct." (φ̇: watch the two separate ½ factors, one from H,
one from ∂θ/∂ρ(1) = ½√(ρ(2)/ρ(1)).)

> "on (b), are you identifying a bona fide bug in the code, or just a case
> where there hypothetically could have been a bug? and on (c) is this the
> tolerance for the mollification for the single-node target?"

- **(b) Hypothetical, not a bug.** Flipping the sign of *both* equations of
  motion runs the same geodesic backwards in time: mass, H, and 2H = W2 are all
  still satisfied, and the two-node closed-form test uses `abs`. So the Module
  3.1 tests are blind to a global sign flip (which is just the convention
  φ ↦ −φ). It gets pinned in the `log_map` tests, where `m0` is compared to the
  SOCP's `m0` without `abs`. → coverage note, with the downstream cover named.
- **(c) Different tolerance.** The 1e-4 is the H-conservation bound over 200
  RK4 steps; RK4's global error is O(h⁴) ≈ 6e-10 here, so the bound is loose by
  orders of magnitude. → coverage note ("set tolerances from a measured value
  plus margin, and say what was measured"). The mollification tolerances (1%
  extrapolated, 0.5% raw vs. the SOCP) are *accuracy* tolerances on an
  approximate method and were set from measured values. The single-node target
  is not in the test suite; it lives under "Known limits".

---

## 4. Step 3: `src/shooting/ExpLog.jl`, helpers and `exp_map`

**The linear solve** (`solve_weighted_laplacian`, lines 30–34). L = ∇ᵀDiag(κθ)∇
is singular (L·𝟙 = 0). The code solves (L + ππᵀ)φ = b instead:

1. Invertible: ππᵀ𝟙 = π ≠ 0 covers the kernel direction; both terms PSD ⇒ sum
   PD ⇒ `cholesky` is legal (and would throw loudly otherwise).
2. Same solution: left-multiply by 𝟙ᵀ; the L term vanishes, leaving
   (𝟙ᵀπ)(πᵀφ) = Σb = 0 ⇒ πᵀφ = 0 ⇒ Lφ = b.
3. Gauge: πᵀφ = 0 is exactly ⟨φ,1⟩_π = 0, obtained for free.

The hypothesis Σb = 0 is automatic for `momentum_to_potential` (each edge
contributes +κm − κm) and holds for `log_map`'s `π∘(target−ν)` because both
are densities (`log_map` asserts this upstream; the assert on line 31 is a
second line of defense).

> "there is a comment on 'n being at most a few hundred' we should flag this
> in the review since we will want to empirically push on how big we can make
> these graphs. additionally, I am wondering why A is wrapped in Symmetric, and
> why it even gets factored to begin with? are we doing repeated solves here?
> [...] it's not as if the factored matrix is being returned."

- **Flag it, but precisely.** This solve runs once per `log_map` (initial
  guess) and once per `exp_map` with a momentum tangent. Dense Cholesky ≈ n³/3
  flops: milliseconds at n = 500, seconds + 200 MB at n = 5000. It is *not* the
  scaling wall. The wall is the ForwardDiff Jacobian in `log_map`: ≈ (n/12)
  full 150-step integrations per Newton step (MA house 4× slower than the
  100-node grid for 1.6× the nodes). Review comment: state the assumption at
  package level, not in a line comment, and name the real bottleneck.
- **`Symmetric`**: A is symmetric in exact arithmetic but can be off in the last
  bit; `cholesky(::Matrix)` checks exact symmetry and would throw. `Symmetric`
  says "read one triangle". Standard idiom, free.
- **Why factor at all**: there is no dense solve without a factorization;
  `A \ b` *is* factor + two triangular solves. The choice is which
  factorization: pivoted LU (2n³/3) vs. Cholesky (n³/3, SPD-exploiting). So
  `cholesky(Symmetric(A)) \ b` is the same work as `A \ b`, done right.
- **Reuse**: the matrix depends on ν through θ(ν), so in a moving-base loop
  nothing is reusable. Within one `log_map` the solve happens once; the Newton
  iterations use a different matrix (the flow-map Jacobian, LU'd fresh each
  iteration, unavoidable). The one reuse opportunity is `analyze_shooting`,
  where the base is fixed across p references and the same Laplacian is
  factored p times; negligible next to the Newton solves. → taste note,
  "optimization available, not needed".

Open from this step (not yet answered): the n = |E| branch of `exp_map`
(lines 73–82; which test graph has n = |E|, does a test exercise the error
path, is refusing to guess the right design?).

---

## 5. Step 4: `log_map` at full depth

> "is this a massive PR? feels like its taking a while to get through"

No: medium-sized, on the large end of what a reviewer accepts unsplit. It feels
slow because you are learning both the review skill and the module's math at
once, and neither repeats.

> "I'm a little confused how we're using Newton's method here. [...] My
> baseline intuition on using Newton's method to optimize a function F is that
> you need its gradient DF and its Hessian D2F [...] That doesn't seem to be
> what we're doing here."

It is Newton for **root-finding** (the Calc 1 version, x ← x − f(x)/f′(x)),
not for optimization. The optimization version is root-finding applied to
f = g′, which is why it needs g″ (the Hessian). Here we solve an equation
directly, so one derivative suffices.

- Flow map Φ: φ₀ ↦ ρ(1; ν, φ₀), computed by integrating the ODE. Solve
  F(φ₀) = Φ(φ₀) − target = 0: n equations, n unknowns.
- Newton: F(φ₀+δ) ≈ F + Jδ, J = ∂Φ/∂φ₀ (n×n: "how does the endpoint at node i
  move when the initial potential at node j is nudged"); solve Jδ = −F
  (line 47, `δ = -(J \ F)`, the analogue of dividing by f′).
- J is not a formula; ForwardDiff pushes dual numbers through the whole RK4
  integration and returns J exact to machine precision ⇒ quadratic convergence
  to 1e-9 (finite differences would stall on noise).
- Newton only converges from nearby ⇒ line search (try α = 1, halve until ‖F‖
  decreases; lines 53–67) and a good initial guess (the linearized geodesic,
  line 20). The 2–4 iteration counts are the signature of quadratic convergence
  from a good start.
- **Two redundancies** make the full J singular: adding a constant to φ₀ does
  nothing (gauge), and Φ always outputs a density (one output constraint). The
  code uses n−1 unknowns z with φ₀(n) fixed by ⟨φ₀,1⟩_π = 0
  (`_reduced_to_potential`) and drops one residual component (`F_reduced`);
  the reduced (n−1)×(n−1) J is generically invertible.
- Optimization view for comparison: minimizing ½‖Φ−target‖² by Gauss–Newton
  gives the step (JᵀJ)⁻¹JᵀF, which for square invertible J equals J⁻¹F. Same
  step; root-finding is just the cheaper way to write it.

> "I guess what I'm hung up on is how this relates to the objective function
> of the variance minimization problem."

Two stacked problems; shooting replaces the **inner** one.

- **Outer**: J(ν) = Σλᵢ W²(ν, refᵢ), minimized over ν. Its Riemannian gradient
  is −2 Σλᵢ log_ν(refᵢ). Synthesis descends it; analysis solves ∇J(ν) = 0 for
  λ (the Gram QP). Everything upstream needs one primitive: log_ν(μ) and
  W²(ν, μ) for one pair. `discrete_transport`, `geodesic_socp`, `log_map` are
  interchangeable providers of it.
- **Inner**: W²(ν, μ) = min over paths of the Benamou–Brenier action
  ∫2H dt subject to continuity. CP and the SOCP minimize over the whole
  space-time path and read log_ν(μ) off as the momentum at t = 0.
- **Shooting**: the action's Euler–Lagrange equations *are* Hamilton's
  equations for H, so minimizers are Hamiltonian trajectories, each determined
  by (ν, φ₀). The only unknown is which φ₀ lands on μ: n boundary conditions,
  enforced by Newton. **"Minimize over paths" becomes "integrate the
  optimality conditions and solve for the initial condition that meets the
  end."** Newton is not minimizing anything; the minimization was done
  analytically when Hamilton's equations were written down.
- Outputs: log_ν(μ) *is* the φ₀ solved for (m₀ = θ(ν)∇φ₀ in momentum form);
  W² = 2H(ν, φ₀) because geodesics have constant speed, so ∫2H dt = 2H(t=0).
- Link to PR #22: ∂W²(ν,μ)/∂ν = −2φ₀ (mod gauge). The SOCP's endpoint dual is
  ∂W²/∂ν by the meaning of a multiplier; shooting's φ₀ is the initial velocity
  potential; Hamilton–Jacobi says they are the same object. `log_map` is the
  derivative of one term of the variance, which is why the variance never
  appears in its code.

> "so does the hamiltonian shooting just replace the heuristic approaches in
> our paper? like if I specify a family of weights and references, and I
> specify I want to synthesize a barycenter via Hamiltonian Shooting, are we
> still doing intrinsic gradient descent?"

Yes: shooting replaces the inner solver, not the outer algorithm.

- **Replaces**: the per-reference Chambolle-Pock call inside each WGD
  iteration. Same gradient, faster, no time-discretization error. Outer loop
  unchanged: same step size, stopping rule, and plateau behavior (MA house:
  descent stalled at J = 11.80 vs. optimum 11.31). Cheaper iterations, not
  fewer.
- **Improves within descent**: `exp_map` allows ν ← exp_ν(−h∇J), a geodesic
  retraction instead of the paper's linearized step ν − h·div(Σλᵢmᵢ). Proper
  Riemannian GD, no "left the positive cone, halve and retry"; still first
  order. The spec's warm-start cache (3.4) exists for exactly this loop.
- **Replaces descent entirely**: `barycenter_socp`, not shooting. One convex
  program, global optimum, no outer loop; scales badly (spec §2.4 fallback on
  the horizon).
- Synthesis menu: WGD+CP (`barycenter()`, the paper); WGD+shooting with
  geodesic retraction (not built; would be `barycenter_shooting`); joint SOCP
  (`barycenter_socp`). Analysis menu: `analysis`, `analyze_socp`,
  `analyze_shooting`, same Gram matrix from different solvers, with the #22
  caveat that each recovers cleanly only barycenters synthesized in its own
  discretization.
- Open question for the paper's story: build the shooting-based descent as
  "the paper's algorithm done right", or treat descent as a baseline and let
  shooting's job be analysis + the log/exp primitives.

### The `log_map` code, block by block (the "wall of text", to return to)

Block 1, setup/reduction (`_reduced_to_potential`, lines 4–17):
- Gauge solved for node n; divides by π(n). Fine for irreducible chains (all
  π > 0, comparable) but unstated. → taste: pick argmax π or document.
- Line 7 asserts both endpoints are densities (the upstream check for the
  Laplacian solve's Σb = 0).
- Line 16 drops the last residual component (exact up to integrator error,
  since Σπ(x)F(x) = 0); line 17's norm keeps all n. Two different residual
  objects; harmless, but notice it.

Block 2, initialization/damping (lines 19–41):
- Line 20: linearized geodesic Lφ₀ = π∘(target−ν), used once.
- Lines 29–40 halve z until the first shot survives the floor; k = 12 errors
  before halving again (correct, took a re-read → mild taste).
- **Finding #2**: `catch err; err isa ErrorException || rethrow()` catches the
  integrator's floor error, but `ErrorException` is what *every* `error(...)`
  throws. A genuine bug inside `hamiltonian_flow` that called `error` would be
  interpreted as "hit the floor" and answered by halving the step, pursuing a
  wrong answer instead of surfacing a crash. Fix: a dedicated exception type
  for floor violations. Same pattern at lines 55–60.

Block 3, Newton loop (lines 43–72):
- Absolute tol on ‖F‖_π (defensible, densities are O(1); relative would be
  more robust). Non-convergence after `maxiters` is a hard error (loud, good).
- Line 46: ForwardDiff's internal F evaluation is discarded; one wasted
  integration per iteration ≈ 1/(n/12 + 2) of the cost. Note only.
- Line 47: a singular J throws `SingularException` (not caught; loud but
  unhelpful message). No test exercises it; no known case produces it.
- Lines 53–67: Armijo-style backtracking, 12 halvings (α ≥ 1/4096); a
  floor-hit inside the line search counts as a failed step and shortens α
  (this is what let the corner-to-corner case converge).
- **Mechanism worth knowing**: the floor guard *bisects* steps near the
  boundary, so F is piecewise-smooth in z there; ForwardDiff differentiates
  the branch taken, so J is exact for that branch but the function jumps
  between branches. Newton stalls at the jump scale. This is almost certainly
  the ε = 1e-4 stall at residual 6e-7 on Julia 1.10 and the ε = 1e-5 failure.
  The "target may be too far" message misattributes the cause. → description
  gap (+ future fix: smooth barrier instead of bisection).

Block 4, outputs (lines 74–77): m₀ = θ(ν)∇φ₀, W2 = 2H, per spec. This is where
the `m0`-vs-SOCP test finally pins the global sign.

**Question 1 resolved: approve with a non-blocking request.** Nothing in
`hamiltonian_flow`/`_rk4_step` calls `error` today, so the only
`ErrorException` reachable from `shoot` is the floor error; the code is
behaviorally correct. Even if a future flow bug threw one, the worst case is a
misleading error message or wasted halving, never a wrong result, because any
returned φ₀ has passed ‖ρ(1)−target‖_π < tol on a trajectory that actually
integrated. Fix is ~10 lines (`struct PositivityFloorError <: Exception`,
thrown at Hamiltonian.jl:128, matched at ExpLog.jl:157/182). Review wording:
"approve; non-blocking request to introduce a dedicated exception type, since
the fix is ten lines and it removes the only way a future flow bug could hide."
(Production vs. research: the correctness bar is the same; what differs is who
pays for non-fatal issues like misleading messages.)

**Question 2 resolved: a real test-quality finding.** Deleting the damping loop
(ExpLog.jl:153–164) makes the first `shoot(z)` throw with nothing to catch it
(the line-search catch at :181 is later in the function), so `r` at
runtests.jl:845 is never bound and the testset records an *Error*, not a failed
assertion. The feature is guarded, but only because removal happens to crash;
the three assertions (:846–848) check convergence, not damping. And nothing
verifies the premise in the comment (:838, "the linearized guess overshoots
here"): change the grid size or concentration weight and the test could pass
while no longer exercising damping. Fix: assert the premise with public
functions,
```julia
φ0_lin = solve_weighted_laplacian(G, ν, π .* (μ .- ν))
@test_throws ErrorException integrate_hamiltonian(G, ν, φ0_lin; nsteps=150)
```

---

## 5b. Step 5: `log_map_mollified` and `analyze_shooting`

`analyze_shooting` (10 lines): one `log_map` per reference, shared Gram QP
(`potential_gram_qp`, refactored out of `analyze_socp`; existing `analyze_socp`
tests cover behavioral equivalence). No positivity pre-check of its own (relies
on `log_map`'s asserts; a late failure wastes earlier solves — taste). No
fallback to the SOCP on a failed reference: "fall back to geodesic_socp" is
advice to the caller, not behavior. Acceptable if the docstring says so plainly.

`log_map_mollified`: the uniform density is the constant 1 (⟨1,π⟩ = 1), so
(1−ε)ρ + ε·1 is a density ✓. Levels warm-start from the previous level's φ₀ (at
a different base point; a good guess, not exact). Least-squares fit
W(ε) ≈ W₀ + a√ε per spec. Both probes showed decay *faster* than √ε (successive
differences shrank ~7× per decade vs. ~3.2× predicted), and the fit
over-corrected by 0.3% / 0.5% while the raw smallest-ε value was closer.
Returned `φ0`/`m0` live at ν_ε, not ν (docstring says so).

> "keep the fit as spec'd but document the caveat, we'll want to know about
> how numerical evidence performs vs expected theory when writing up the paper"

**Decision: keep the √ε fit; add the observed-vs-predicted decay to the docs
and PR description** (description gap #4).

> "is there a warning that prints for this kind of scenario?" [implicit
> mollification]

No, and none is needed: mollification is never implicit. `log_map` and
`analyze_shooting` *error* on non-interior data ("log_map requires strictly
positive endpoints"); `log_map_mollified` is opt-in by name and has no library
callers (tests only). Its one `@warn` fires when an ε level is skipped, which
is the right place. **Decision: the docstring is sufficient for the ν_ε naming.**
Review items: state in "Known limits" that non-interior data errors rather than
being routed; add "use geodesic_socp or log_map_mollified" to that error
message.

Coverage notes: the skip-a-level path is exercised only by accident (Julia
1.10); `issorted(r.Ws)` asserts a monotonicity that is plausible, not proven.

## 5c. Steps 6–7: tests as adversary; benchmark and wiring

Main `log_map` testset (runtests.jl:798–827): `iters ≤ 8` is protected by the
function's own exit condition, not the assertion; round trips are
self-consistency; the "warm start" restarts from the exact solution (:809,
also :880 in `analyze_shooting`) so tests little; the O(h) block (:813–820)
*does* test a rate and is the pattern finding #1 should copy; :824 (the −2
relation, rtol 5%) is the line that pins the global sign (a flip gives 200%
error, a factor −1 gives 50%). Random pairs are mild (rand+0.5); the far-apart
regime is covered only by the damping test; 60 benchmark pairs are the real
convergence evidence and are not in CI.

> "1. it definitely should matter, the gram matrix entries are defined by
> inner products of vectors in the tangent space at the reference measure,
> ergo if θ is wrong, the inner products are wrong."

Right mechanism; the subtlety is why no test notices: with an exactly
stationary synthesized target, Σλᵢ∇φᵢ = 0 makes λ the QP minimizer under *any*
positive weighting, so every recovery test here and in #22 is blind to κθ vs.
θ vs. 1. The weighting matters exactly where the paper uses analysis:
non-stationary (real) targets, and cond(A). Test: `potential_gram_qp` with
`return_system=true` on the triangle vs. the hand-assembled Riemannian inner
product via dense `metric_tensor`. → upgraded to **requested change #3**.

Small testsets: "π∘ρ̇ == Lφ" is a real cross-check (Laplacian assembly is
independent of `D`); the `t=0.5` `exp_map` check compares two different step
sizes at rtol 1e-10 (taste: `nsteps=50`); two-node closed form is sign-blind
but covered by :824; guards fine.

Step 7: all 11 new exports documented in `api.md`; the two lines removed from
`analyze_socp` are exactly `potential_gram_qp`; benchmark is seeded and writes
only to cwd. `potential_gram_qp` is named in a docstring but not exported
(taste).

> "2. No I think we can basically approve this PR, and supply some notes about
> future things to keep an eye on / clarify as we iterate on the library"

**Verdict: approve with three small requested test changes** (#1 rate check,
#2 damping premise, #3 Gram weighting); exception type as non-blocking
follow-up; description edits; the rest as notes. Drafted in `pr23_review.md`.
GitHub does not allow approving one's own PR: post as a PR comment, then merge.

## 6. Running tally (to become the posted review)

**Findings**
1. `analyze_shooting` test asserts a small error at a single N (80) while the
   description claims an O(h) rate; test two N values and check the error
   decreases.
2. The damped-initialization test (runtests.jl:837–849) guards the feature only
   via an incidental crash and does not verify its own premise; add an
   assertion that the undamped linearized guess actually violates the floor.

3. `exp_map` on an n = |E| graph with a wrong-length tangent reports "kind
   cannot be inferred" (ExpLog.jl:78–79) instead of "wrong length", and after
   passing `kind` explicitly hits a bare `@assert` (:85). Reorder: check length
   against both n and |E| first. Untested (runtests.jl:791 uses the hypercube).
   Small.

4. The Gram matrix's κθ weighting is untested by any recovery test (exactly
   stationary targets make λ̂ weighting-invariant). Unit-test
   `potential_gram_qp`'s `A` against the dense Riemannian inner product on the
   triangle. Covers both backends. → requested change #3.

**Non-blocking request**
- `log_map` catches `ErrorException` broadly (ExpLog.jl:157, 182); introduce a
  dedicated floor-violation exception type (~10 lines). Cannot cause a wrong
  result today; removes the only way a future flow bug could hide.

**Description gaps**
1. Spec 3.4's warm-start *cache* is not implemented; only `φ0_init` is.
2. The scaling limit is the ForwardDiff Jacobian in `log_map` (~n/12
   integrations per Newton step), not the dense Laplacian solve.
3. Near-boundary stalls come from the bisecting floor guard making F
   piecewise-smooth, not from generic "stiffness"; the error message
   misattributes it.
4. The mollification error decayed faster than the spec's √ε model in both
   probes (fit over-corrects 0.3–0.5%); say so, for the paper's
   numerics-vs-theory discussion.
5. "Known limits" should state that non-interior data errors in `log_map` /
   `analyze_shooting` rather than being routed to mollification; and the error
   message should name `geodesic_socp` / `log_map_mollified` as the options.

**Coverage notes**
1. Module 3.1 tests are blind to a global sign flip of the flow; cover comes
   from the `m0`-vs-SOCP comparison in the `log_map` testset.
2. H-conservation tolerance 1e-4 is loose by several orders vs. RK4's O(h⁴).
3. No test exercises a singular Jacobian in `log_map` (no known trigger).
4. `exp_map` with an explicit `kind=:momentum` is untested on an n = |E| graph
   (the path itself is covered on the hypercube).
5. `log_map_mollified`'s skip-a-level path is exercised only by accident
   (Julia 1.10); `issorted(r.Ws)` asserts unproven monotonicity.

**Taste**
0. `kind` exists only to patch length-based inference of potential-vs-momentum
   tangents; wrapper types (`Potential(φ)`, `Momentum(m)`) would remove the
   ambiguity and the finding above. Momenta have no non-test caller today, so
   not worth it yet.
1. "n at most a few hundred" belongs in package-level docs, not a line comment.
2. Gauge fixed at node n divides by π(n); pick argmax π or document the
   positivity assumption.
3. `analyze_shooting` refactors the same Laplacian p times; a cached
   factorization would remove it (negligible cost today).
4. The k = 12 exit in the damping loop took a re-read.

**Done.** Review drafted in `pr23_review.md`; post as a PR comment, apply the
three test changes, merge.
