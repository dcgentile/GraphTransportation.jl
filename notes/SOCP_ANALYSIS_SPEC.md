# SOCP-native analysis (Module 4) — problem spec & investigation plan

Companion to `spec.txt` (Module 4, "Analysis pipeline"). Written for a fresh
session with no memory of the investigation that produced it — everything
needed to pick this up is below, including exact file paths and cached data.

## 1. Background

`spec.txt` Module 4 specifies coordinate recovery as: for a `target` measure
and reference measures `refs`, get each geodesic's potential `φ_i` from
`target` to `ref_i`, build a Gram matrix

```
A_ij = ½ Σ_{x,y} θ(target(x),target(y)) ∇φ_i(x,y) ∇φ_j(x,y) Q(x,y)π(x)
     = Σ_e θ(target)_e (∇φ_i)_e (∇φ_j)_e κ_e
```

and solve `min_{λ∈Δ^{p-1}} λᵀAλ` for the recovered weights `λ̂`.

The actual implementation, `analyze_socp` (`src/socp/Analysis.jl`), does
**not** do this. Its docstring instead builds the Gram matrix from
**momenta** (`m0`, the initial momentum of each geodesic, from
`geodesic_socp`), reusing the exact same `solve_barycentric_coordinates_qp`
QP machinery that the pre-existing, dissertation-validated `analysis()`
function uses for Chambolle-Pock (CP) geodesics — on the stated grounds that
`m = θ(target)∘∇φ` makes momenta and potential-gradients equivalent, and
that both `discrete_transport` and `geodesic_socp` are already known to
produce the same `m` (Module 1's §1.3.2 gate). Session memory from the
branch that built this (not the code/docstring itself, so unverified —
worth confirming if it matters) additionally records that extracting `φ`
directly from JuMP's continuity-equation duals was tried first and set aside
because "the naive ratio wasn't a clean constant (needs the RSOC constraints'
own duals too)". Everything in §2 below shows the claimed momentum/potential
equivalence does *not* hold well enough in practice, at least at coarse `N`
on graphs bigger than the toy cases Module 1's own gate was checked on.

This spec is about why that substitution breaks down in practice, and what
to try instead.

## 2. What we found (this session)

All of this is reproducible from `src/experiments/` on the `hamiltonian-shooting`
branch (SOCP module — `MarkovGraph`, `barycenter_socp`, `geodesic_socp`,
`analyze_socp` — already present in this working tree).

### 2.1 MA house graph (160 nodes), descent vs SOCP

Script: `src/experiments/MassachusettsSOCPComparison.jl`. Cache:
`src/experiments/ma_house_socp_comparison.jld2` (has `bar_descent`, `ν_socp`,
`rc_descent`, `rc_socp`, and more — reload this rather than recomputing;
`bar_descent` alone took ~58 minutes to synthesize at production tolerance).

Setup: 4 reference measures (geographically concentrated at MA house
district indices 5, 50, 100, 145, weight 10), `λ = [0.4, 0.3, 0.2, 0.1]`,
`N=2` (geodesic time-steps), descent at `tol=1e-8, geodesic_tol=1e-10,
maxiters=8192` (matches the paper figure's production settings).

| quantity | value |
|---|---|
| `J_descent` (`Σλᵢ𝒲²(refᵢ,bar_descent)`, evaluated via `geodesic_socp`) | 11.7993 |
| `J_socp` (global optimum) | 11.308 |
| `‖ν_descent − ν_socp‖_π` | 0.435 |
| recovered λ̂ from `bar_descent` (via `analysis()`) | `[0.400,0.301,0.200,0.098]`, rel. err **0.38%** |
| recovered λ̂ from `ν_socp` (via `analyze_socp()`) | `[0.294,0.408,0.199,0.099]`, rel. err **27.6%** |

So: `ν_socp` is the *better* barycenter (lower `J`, and it's the global
optimum by construction) but recovers coordinates far *worse*. `bar_descent`
is a plateaued, non-fully-converged WGD solution (its own loss curve is
visibly flat well above `J_socp` from iteration ~30 onward) yet its recovered
coordinates are excellent.

### 2.2 Isolating the cause: solver vs. target point

We cross-tested both geodesic solvers (CP's `discrete_transport` and SOCP's
`geodesic_socp`) at both target points, building the Gram matrix each way
(`GraphTransportation.solve_barycentric_coordinates_qp`, not exported):

| target | solver for tangent vectors | recovered λ̂ | rel. err | `cond(A)` |
|---|---|---|---|---|
| `bar_descent` | CP (`discrete_transport`) | `[0.400,0.301,0.200,0.098]` | 0.38% | 4090 |
| `bar_descent` | SOCP (`geodesic_socp`) | `[0.399,0.304,0.201,0.096]` | 0.98% | 1170 |
| `ν_socp` | CP (`discrete_transport`) | `[0.296,0.404,0.199,0.101]` | 26.8% | 22.3 |
| `ν_socp` | SOCP (`geodesic_socp`) | `[0.294,0.408,0.199,0.099]` | 27.6% | 21.9 |

**The solver barely matters; the target point is everything.** Whichever
solver computes tangent vectors, `bar_descent` recovers well and `ν_socp`
doesn't. This rules out "SOCP's geodesic solver is unreliable" as the
explanation — both solvers agree `ν_socp` is a hard point to analyze.

We also checked whether `analyze_socp`'s *independent* re-solve of each
reference's geodesic (from `ν_socp`) disagrees with the momenta
`barycenter_socp`'s own *joint* solve already computed internally
(`geodesics` field on its return value) — they agree to ~5e-5 relative, i.e.
essentially exactly. So the discrepancy is not a joint-vs-independent-solve
artifact either.

### 2.3 The actual mechanism: stationarity-convention mismatch

`solve_barycentric_coordinates_qp` recovers `λ̂` by minimizing
`x'Ax = ‖Σᵢ xᵢ mᵢ‖²_g` over the simplex — i.e. it finds whichever weights
make the tangent vectors cancel best. This only recovers the *true* `λ`
accurately if `Σᵢ λᵢ mᵢ(target) ≈ 0` **already**, i.e. if `target` is a
near-stationary point *for the true λ, in this specific tangent-vector
convention*.

We checked this directly: evaluate `λ'Aλ` (residual at the *true* λ) against
`λ̂'Aλ̂` (the QP's actual achieved minimum):

| target (tangent convention) | `λ'Aλ` | `λ̂'Aλ̂` | ratio | `diag(A)` (scale reference) |
|---|---|---|---|---|
| `bar_descent`, CP | 17.91 | 17.62 | **1.02** | `[93K, 58K, 81K, 111K]` |
| `ν_socp`, SOCP | 7068.9 | 3631.5 | **1.95** | `[152K, 48K, 85K, 136K]` |

`diag(A)` (the individual `‖mᵢ‖²` terms) is comparable magnitude in both
cases (~1e5), ruling out a units/scaling artifact. The real signal is the
*ratio*: at `bar_descent` the true λ is essentially already optimal for the
QP (2% off); at `ν_socp` the true λ's residual is **95% larger** than the
QP's actual minimum — there is a substantially different `λ̂` that fits the
observed tangent-vector geometry much better than the true `λ` does.

**Why this happens:** `bar_descent` was produced by `barycenter()`
(`src/Barycenters.jl`), whose WGD loop drives `Σᵢ λᵢ mᵢ(ν) → 0` — using
`discrete_transport`'s own momentum convention — as its literal stopping
criterion. It is *by construction* close to a zero-residual point for the
true λ in exactly the convention `analysis()` checks. `ν_socp` is optimal in
a *different* sense: KKT-stationarity of `barycenter_socp`'s joint SOCP
(shared-`ν` constraint across per-reference RSOC/mean-cone blocks). That is
supposed to be the same continuum condition as `Σλᵢmᵢ=0`, but at `N=2` the
two discretizations (CP's primal-dual operator splitting vs. SOCP's direct
space-time program) just don't agree numerically to anywhere near the
precision needed — and this gap is invisible to `J` (the objective value),
which is a completely different quantity from "is this point stationary in
the momentum-Gram sense."

### 2.4 Generalization: grid graphs, scale, and reference proximity

Scripts: `src/experiments/SOCPGridConsistency.jl` (cache:
`socp_grid_consistency.jld2`) and `src/experiments/SOCPProximityStudy.jl`
(cache: `socp_proximity_study.jld2`). Both are **pure SOCP round-trip**
tests — synthesize via `barycenter_socp` at known λ, recover via
`analyze_socp`, no descent involved.

- On `k×k` grid graphs (`V=25..100`), 4 references at grid corners,
  `λ=[0.4,0.3,0.2,0.1]`, `N∈{2,5,10,20}`: worst-case error across all 24
  cells was **9.7%** (V=100, N=2) — far short of MA house's 27.6%. Error
  drops smoothly and monotonically with `N` at fixed `V` (no swap/instability
  anywhere in this sweep). At fixed `N∈{2,5}`, error grows mildly and
  smoothly with `V`. At `N∈{10,20}` there's a non-monotonic *dip* at V=49
  (best-case error) with error rising again for larger `V` — unexplained,
  possibly a resonance between `N` and corner-to-corner path length, not
  investigated further.
- Reference proximity matters a lot: fixing 2 of 4 references at distant
  corners and sweeping the separation `d` of the other two from the original
  corner-to-corner distance down to `d=1` (adjacent nodes), on both V=49 and
  V=100, at `N=2`: error rises from ~7-10% (max separation) to **~18.7%**
  (`d=1`, both grid sizes) — consistently the worst point in every sweep,
  after a mild non-monotonic dip at intermediate separation.
- Even the worst controlled case (18.7%) doesn't fully reach MA house's
  27.6%, suggesting MA house's real (non-grid, irregular) graph topology
  compounds with reference proximity rather than being explained by either
  alone.

## 3. Why we need a different analysis approach

The momentum-based Gram matrix (borrowed from CP's `analysis()`) checks
stationarity in a convention that `barycenter_socp`-synthesized points are
not guaranteed to satisfy, except in the continuum limit. This is not a bug
in the QP solver or in `geodesic_socp`'s momentum computation (both checked
directly above) — it's a **convention mismatch between SOCP synthesis and
CP-native analysis**, and it gets worse at coarse `N` and with closer/more
similar reference measures, and worse still on irregular real-world graphs.
Refining `N` masks it (MA house: 27.6%→7.9% from N=2→10) but doesn't fix it,
costs significant compute (`barycenter_socp` scales worse than linearly in
both `V` and `N` — up to ~13s per solve at V=100, N=20 in our sweep), and we
have no principled way to know how much `N` is "enough" without already
having ground truth to check against.

The fix implied by `spec.txt`'s own original design: build the Gram matrix
from the SOCP's **own** dual/potential structure instead of borrowing CP's
momentum convention. This was tried once and abandoned as too fiddly
(needs RSOC duals folded into the continuity-constraint duals to get a clean
`φ`) — but given what we now know about *why* the momentum approach fails,
finishing that derivation looks like the theoretically-correct fix, not just
an alternative.

## 4. What we hope to see

1. A `:socp`-native analysis path (extracting `∇φ` from `barycenter_socp`'s
   own solve, per spec.txt's formula) whose round-trip recovery error, when
   synthesizing and analyzing entirely within the SOCP pipeline, matches
   descent's demonstrated self-consistency (sub-1% on MA house) rather than
   the 27%+ currently seen at `N=2`.
2. Recovery quality that is much less sensitive to `N` — i.e., the fix
   should come from using the *right* stationarity convention, not from
   brute-force refining the discretization until the wrong convention's
   error becomes small by accident.
3. A clean account of why: recovered `λ̂` from the native Gram matrix should
   coincide almost exactly with `λ'Aλ ≈ λ̂'Aλ̂` (residual ratio ≈ 1, the same
   near-1.0 signature `bar_descent` already shows), evaluated at whatever
   point `barycenter_socp` actually produces, *without* needing that point to
   also happen to satisfy CP's convention.
4. (Stretch) If `barycenter_socp`'s own JuMP model exposes usable dual
   sensitivities on the shared-`ν` / per-block constraints directly, it may
   be possible to skip a separate analysis QP entirely for the special case
   where `target` is known to have come from `barycenter_socp` itself (as
   opposed to `analyze_socp`'s more general "arbitrary target measure" case,
   which still needs the potential-based Gram matrix from freshly-solved
   geodesics).

## 5. Proposed approach (sketch — expect to revise during implementation)

1. **Extract potentials, not momenta.** In `geodesic_socp`
   (`src/socp/Geodesic.jl`), the continuity-equation constraint
   `(ρ[:,t+1]-ρ[:,t])/h + divm == 0` is added per time-step `t` inside
   `_geodesic_block!`. Its dual at `t=1` is a candidate for `φ_i` (per
   `spec.txt` §3.3's own note: "Optional high-quality init: run Module 1 at
   coarse N=5 and read off φ from continuity-equation duals" — so this dual
   extraction is already known to work for *something*; the open question is
   whether it's clean enough alone for the Gram matrix, which the abandoned
   attempt says it wasn't).
2. **Fold in the RSOC duals.** The mean-cone (`ϑ² ≤ ρ̄ₓρ̄ᵧ`) and
   action-epigraph (`m² ≤ ϑw`) `RotatedSecondOrderCone` constraints in
   `_geodesic_block!` also involve `ρ` at each `t`, so the *full* KKT
   stationarity condition on `ρ[:,t]` mixes continuity-constraint duals with
   these RSOC duals. Write out the Lagrangian stationarity condition for
   `ρ[:,t]` by hand (paper/derivation, not code) to find the exact
   combination that gives a clean `∇φ`, before writing any Julia.
3. **Sign/scale calibration.** `spec.txt`'s own "Known pitfalls" checklist
   flags this explicitly: *"Duals from JuMP have solver-dependent sign;
   calibrate against θ(ρ̄)∘∇φ = m on a solved instance."* Do this calibration
   on a 2-node closed-form instance (see Experiment 1 below) before trusting
   the sign/scale on anything larger.
4. **Assemble the Gram matrix** per `spec.txt`'s formula,
   `A_ij = Σ_e θ(target)_e (∇φ_i)_e (∇φ_j)_e κ_e`. Check whether a
   compact-edge-vector `θ(target)` helper already exists or needs adding
   alongside `MarkovGraph`'s existing `graph_gradient`/`graph_divergence`.
5. **Reuse `solve_barycentric_coordinates_qp` unchanged** — it's already
   solver/convention-agnostic (just takes `tangent_vectors` + a metric), so
   no changes needed there.

## 6. Experiments to validate

1. **2-node closed-form calibration gate.** Before anything else: check
   `geodesic_socp`'s dual-derived `φ` against the closed-form two-node
   geodesic already used to gate Module 1 (`test/runtests.jl`, testset
   `"geodesic_socp vs two-node closed form (SOCP Module 1, spec §1.3.1)"`,
   line ~411 — reuse its `ρ(r) = [1-r, 1+r]` setup and closed-form distance).
   Confirms sign/scale before scaling up. Cheap, deterministic, should fail
   loudly and immediately if the convention is wrong.
2. **MA house re-test.** Reload `ma_house_socp_comparison.jld2`'s cached
   `ν_socp` (no need to resynthesize — that part is unchanged) and rerun
   analysis with the new native Gram matrix at `N=2`. Target: recovery error
   drops from 27.6% to something comparable to `bar_descent`'s 0.38%,
   *without* needing to go to `N=10`.
3. **Residual-ratio check.** Recompute `λ'Aλ` vs the QP's achieved minimum
   (§2.3's diagnostic) using the new native `A`; target ratio ≈ 1.0, not 1.95.
4. **Grid consistency re-sweep.** Rerun `SOCPGridConsistency.jl`'s 24-cell
   sweep (same cached grids/references) with native analysis substituted;
   check errors are uniformly small and much less `N`-dependent than the
   current momentum-based results.
5. **Proximity re-sweep.** Same for `SOCPProximityStudy.jl`'s 32-cell sweep;
   check whether native analysis is also less sensitive to reference-pair
   proximity, or whether that turns out to be a separate, still-present
   effect even with the convention fixed.
6. **Full descent-vs-SOCP re-comparison.** Once native analysis passes 2-5,
   rerun `MassachusettsSOCPComparison.jl` end to end (reusing the cached
   `bar_descent`, only resynthesizing/re-analyzing the SOCP side) and confirm
   the "SOCP recovers worse than descent despite lower J" asymmetry is gone
   or substantially reduced.

## 7. Risks / open questions

- The RSOC-dual-folding derivation (step 2 above) may turn out exactly as
  fiddly as the abandoned attempt found it. Time-box Experiment 1; if the
  2-node calibration gate doesn't converge on a clean, stable sign/scale
  convention in reasonable time, fall back to documenting this as a known
  limitation of the SOCP analysis pipeline at coarse `N`, with "refine `N`"
  as the practical mitigation, rather than continuing to chase the native
  derivation indefinitely.
- Need to confirm what JuMP/Clarabel actually expose for
  `RotatedSecondOrderCone` constraint duals (`MOI.ConstraintDual` support
  for this cone type with this solver) before assuming the derivation in
  §5 is implementable at all as described.
- Worth checking whether this is specific to the discrete-transport/Erbar et
  al. momentum-vs-potential duality, or a more general "synthesis and
  analysis must share a discretization convention" lesson that will also
  bite the `:shooting` backend (Module 3/4, log/exp-map based analysis) once
  it's built — that backend uses yet another discretization (RK4 Hamiltonian
  flow) that hasn't been checked against either convention here.

## 8. Results (2026-09-15)

Implemented on `hamiltonian-shooting` in three commits (`8a50c10`, `dcc5baa`,
and the experiments commit after them). Everything in §4 was met.

**Calibration (§6.1).** JuMP/Clarabel's dual of the endpoint constraint
`ρ[:,1] .== ρA`, divided by `π`, is exactly `∂W_h²/∂ρA` in the π-weighted
pairing: it matches a central finite difference of `geodesic_socp`'s own `W2`
to ~1e-5 and the two-node closed form with O(h) error. No sign flip is
needed. The RSOC-dual folding in §5.2 turned out to be unnecessary: the
endpoint duals *already* are the potential. The continuity-equation duals
`ψ_t` satisfy `m_t = -(1/2h) θ(ρ̄_t) ∇(ψ_t/π)` with a clean constant — the
earlier "not a clean constant" came from pairing with `θ(target)` instead of
the interval-midpoint `θ(ρ̄_t)`. That is also *why* the momentum convention
fails: `m0` lives on the first interval and only approximates the endpoint
potential to O(h).

**MA house (§6.2, §6.3).** Reloading the cached `ν_socp` (N=2):

| target | convention | λ̂ | rel. err | λᵀAλ / max diag(A) |
|---|---|---|---|---|
| `ν_socp` | potential (new) | `[0.4000, 0.3000, 0.2000, 0.1000]` | **1.3e-5** | 3.7e-8 / 140 |
| `ν_socp` | momentum (old) | `[0.294, 0.408, 0.199, 0.099]` | 27.6% | 7069 / 1.5e5 |
| `bar_descent` | momentum (old) | `[0.400, 0.301, 0.200, 0.098]` | 0.38% | 58 / 1.1e5 |
| `bar_descent` | potential (new) | `[0.474, 0.215, 0.216, 0.095]` | 20.8% | 3.0 / 144 |

The true λ is a numerical zero of the native Gram form at `ν_socp` (the
residual ratio in §2.3 is no longer meaningful there, since both residuals
are solver noise). The mirror-image row — `bar_descent` recovered badly by
the *potential* convention — confirms the diagnosis: each synthesis method's
output is stationary in its own discrete convention only. Analyzing `ν_socp`
(synthesized at N=2) at N=10 also degrades to 3.5%: synthesis and analysis
must share `N` as well as the convention (§7, third bullet — expect the same
for the `:shooting` backend).

**Grid + proximity re-sweeps (§6.4, §6.5)**, 56 cells, `SOCPNativeResweep.jl`:

| convention | worst rel. err | median rel. err |
|---|---|---|
| momentum (old) | 18.7% (V=100, d=1, N=2) | 1.6% |
| potential (new) | 3.3e-4 | 5.5e-5 |

The native errors are flat in `N`, `V`, and reference separation — the
proximity effect was a symptom of the convention mismatch, not a separate
phenomenon.

**Stretch (§4.4).** `barycenter_socp`'s per-block right-endpoint duals,
divided by `λ[i]π`, give per-geodesic potentials `φ1ᵢ` with
`Σᵢ λᵢ φ1ᵢ = const` on `supp(ν)` to solver tolerance (tested). So for a
target known to come from `barycenter_socp`, `λ` can be read off the joint
solve's duals directly; not wired into an API since `analyze_socp` at the
same `N` already recovers it to ~1e-5.
