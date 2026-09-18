## Review: approve, with three small test changes requested before merge

Verified by hand: the ρ̇ sign against `D`'s definition on the two-node graph (`MarkovGraph.jl:62`, `Hamiltonian.jl:52`); the rank-one Laplacian trick and its gauge (`ExpLog.jl:30-34`); the reduced Newton system and line search in `log_map` (`ExpLog.jl:99-201`); every testset for what a wrong implementation could still pass. The independent evidence (two-node closed form, `W2`/`m0` vs `geodesic_socp` with O(h) convergence, the `−2` relation to the SOCP endpoint dual at `runtests.jl:824`) is sufficient. Nothing found changes a numerical result.

### Requested changes (small, same PR)

1. **`analyze_shooting` test asserts one N.** `runtests.jl:870` checks `atol=1e-2` at N=80 while the description claims O(h). Check two N values and that the error decreases, as `runtests.jl:813-820` already does for `m0`.
2. **Damped-initialization test doesn't verify its premise.** `runtests.jl:837-849` guards the damping loop only because deleting it happens to crash; nothing asserts the linearized guess actually overshoots on this input, so a change to the grid size or concentration weight could silently stop exercising the feature. Add:
   ```julia
   φ0_lin = solve_weighted_laplacian(G, ν, π .* (μ .- ν))
   @test_throws ErrorException integrate_hamiltonian(G, ν, φ0_lin; nsteps=150)
   ```
3. **The Gram matrix's κθ weighting is untested.** Every recovery test (here and in #22) uses an exactly stationary synthesized target, where λ is the QP minimizer under *any* positive weighting. Add a unit test on `potential_gram_qp(...; return_system=true)` on the triangle with two hand-picked potentials, asserting `A[i,j]` equals ½ Σ_{x,y} θ ∇φᵢ ∇φⱼ Q π via the dense `metric_tensor`. Covers both backends; matters for non-synthetic targets and for `cond(A)`.

### Non-blocking request

- `log_map` catches `ErrorException` broadly (`ExpLog.jl:157`, `:182`). Correct today (nothing in the flow calls `error`), and cannot produce a wrong result (every return has passed the residual check on an integrated trajectory), but a future `error` inside the flow would be treated as a floor hit. A `PositivityFloorError <: Exception` thrown at `Hamiltonian.jl:128` and matched at the two sites is ~10 lines. Follow-up is fine.

### Description gaps (edit the PR text)

- Spec 3.4's warm-start *cache* is not implemented; only `φ0_init` is.
- The scaling limit is the ForwardDiff Jacobian (~n/12 full integrations per Newton step), not the dense Laplacian solve; say so and move the "n at most a few hundred" assumption to package-level docs.
- Near-boundary stalls come from the bisecting floor guard making F piecewise-smooth in z (ForwardDiff differentiates the branch taken; Newton stalls at the jump scale). That, not generic stiffness, is the ε ≤ 1e-4 failure; the "target may be too far" message misattributes it.
- The mollification error decayed faster than the √ε model in both probes (fit over-corrects 0.3–0.5%; raw smallest-ε closer). Keep the spec's fit, but record this for the paper's numerics-vs-theory discussion.
- State that non-interior data *errors* in `log_map`/`analyze_shooting` rather than being routed to mollification; add `geodesic_socp` / `log_map_mollified` to that error message.

### Coverage notes (no action)

- Module 3.1 tests are blind to a global sign flip of the flow; cover is `runtests.jl:824` (worth a comment there saying so).
- H-conservation tolerance 1e-4 is loose by several orders vs. RK4's O(h⁴); set from a measured drift plus margin.
- "Warm start" tests restart from the exact solution (`:809`, `:880`); a perturbed target would test the feature.
- `exp_map` `t=0.5` check compares 100 steps of 0.005 to 50 of 0.01 at rtol 1e-10; use `nsteps=50`.
- Untested: singular Jacobian (no known trigger); `kind=:momentum` on an n=|E| graph; the skip-a-level path in `log_map_mollified` (hit only on Julia 1.10).

### Taste

- `exp_map` on n=|E| with a wrong-length tangent reports "kind cannot be inferred" (`ExpLog.jl:78`) instead of "wrong length"; reorder the checks. Longer term, `Potential(φ)` / `Momentum(m)` wrapper types would remove `kind` and the branch; not worth it until momenta have a non-test caller.
- Gauge fixed at node n divides by π(n); use argmax π or document the positivity assumption.
- `analyze_shooting` factors the same Laplacian p times; negligible today.
- `potential_gram_qp` is named in `analyze_shooting`'s docstring but not exported, so the rendered docs can't link it.
