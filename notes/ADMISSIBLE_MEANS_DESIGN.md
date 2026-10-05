# Admissible means beyond the geometric mean: design and theory notes

Goal (user, 2026-09-17): let the SOCP and shooting paths use the arithmetic,
harmonic and logarithmic means in addition to the geometric mean, so the package can
explore how the choice of mobility `θ` changes geodesics, barycenters and recovered
coordinates. Chambolle-Pock stays geometric-only (its prox operators are hand-derived
for that case); it errors on any other mean.

Everything below was checked numerically where it could be (Clarabel power cones,
the harmonic-mean cone form, log-mean quadrature accuracy); the probe is in the
session scratchpad and is reproduced in the tests plan.

---

## 1. Where the mean enters

The discrete transport metric (Maas 2011; Erbar–Maas 2012; Erbar et al. 2020) is

    ‖∇φ‖²_ρ = ½ Σ_{x,y} θ(ρ(x), ρ(y)) (φ(x) − φ(y))² Q(x,y) π(x)  =  Σ_e κ_e θ_e (∇φ)_e²,

and the action along a path is `∫ Σ_e κ_e m_e² / θ(ρ̄_x, ρ̄_y) dt`. The mean appears in
exactly four places in this codebase:

| where | how | today |
|---|---|---|
| SOCP mean cone (`src/socp/Geodesic.jl`, `_geodesic_block!`) | constraint `ϑ_{e,t} ≤ θ(ρ̄_x, ρ̄_y)`, then `m² ≤ ϑ w` | `[ρ̄x, ρ̄y, √2 ϑ] ∈ RSOC` (geometric) |
| Hamiltonian (`src/shooting/Hamiltonian.jl`) | `H = ½ Σ κ θ (∇φ)²`, `ρ̇ = −div(θ∇φ)`, `φ̇ = −½ Σ ∂₁θ (∇φ)² Q` | `√(st)`, `∂₁θ = ½√(t/s)` hard-coded |
| Gram matrix (`src/core/Analysis.jl`, `potential_gram_qp`) | weights `κ_e θ(target)_e` | `metric_tensor(G, target)` defaults to `geomean` |
| shooting helpers (`weighted_laplacian`, `momentum_to_potential`, `exp_map` momentum branch) | `L_θ = ∇ᵀ Diag(κθ) ∇` | `metric_tensor(G, ν)` default |

The dense CP path already has `metric_tensor(ρ, mean)` with `geomean`, `logmean`,
`logmean_partial_s/t` in `src/core/GraphCalculus.jl`, but `KProjection`/the prox
operators assume the geometric mean, which is why CP is left out.

---

## 2. Admissibility, and which means qualify

An admissible mean (Maas 2011, Def. 2.1 / Erbar–Maas) is `θ: [0,∞)² → [0,∞)`
that is continuous, symmetric, positively 1-homogeneous (`θ(λs, λt) = λ θ(s,t)`),
**concave**, and positive on `(0,∞)²` (normalised so `θ(s,s) = s`). Concavity is what
makes the action `m²/θ` jointly convex (perspective of a convex function), hence what
makes the SOCP a convex program with a certified global optimum and the metric a
genuine (Riemannian, on the interior) length metric. All four means qualify:

| mean | `θ(s,t)` | `∂₁θ(s,t)` | `θ(0,t)` | conic form of `ϑ ≤ θ` |
|---|---|---|---|---|
| arithmetic | `(s+t)/2` | `1/2` | `t/2` (does **not** vanish) | linear |
| geometric | `√(st)` | `½√(t/s)` | 0 | 1 RSOC (current) |
| harmonic | `2st/(s+t)` | `2t²/(s+t)²` | 0 | 1 RSOC (below) |
| logarithmic | `(s−t)/(ln s − ln t)` | `logmean_partial_s` (exists) | 0 | not finitely conic-representable; K power cones via quadrature (below) |

Ordering for all `s,t > 0`: harmonic ≤ geometric ≤ logarithmic ≤ arithmetic. So the
arithmetic mean gives the *smallest* distances and the harmonic the largest.

### Harmonic mean as one rotated second-order cone

`ϑ ≤ 2st/(s+t)` ⟺ `(s+t)ϑ ≤ 2st` ⟺ `(s−t)² ≤ (s+t)(s+t−2ϑ)` (using
`4st = (s+t)² − (s−t)²`). With JuMP's `RotatedSecondOrderCone` convention
`2·x·y ≥ ‖z‖²`, that is

    [ρ̄x + ρ̄y,  ρ̄x + ρ̄y − 2ϑ,  √2 (ρ̄x − ρ̄y)] ∈ RotatedSecondOrderCone()

(verified: maximising ϑ under this constraint returns `2st/(s+t)` to 1e-8 for
`(1,3)`, `(0.2,5)`, `(2,2)`; without the `√2` it returns `1.75` for `(1,3)`, i.e. the
factor-of-2 pitfall that bit Module 1 originally).

### Logarithmic mean by Gauss–Legendre quadrature over power cones

`Λ(s,t) = ∫₀¹ s^α t^{1−α} dα`. With nodes `α_k` and weights `w_k` on `[0,1]`,

    Λ_K(s,t) := Σ_k w_k s^{α_k} t^{1−α_k},

and each term is a power cone: `[ρ̄x, ρ̄y, ϑ_k] ∈ MOI.PowerCone(α_k)` means
`ρ̄x^{α_k} ρ̄y^{1−α_k} ≥ |ϑ_k|`; then `ϑ ≤ Σ_k w_k ϑ_k` is linear. Clarabel supports
`PowerCone` (verified: `max z s.t. [2,3,z] ∈ PowerCone(0.5)` returns `√6` to 1e-8).

Accuracy of `Λ_K` (worst relative error over density ratios `s/t ∈ [1.5, 1000]`):

| K | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| worst rel. err | 1.8e-1 | 1.7e-2 | 8.5e-4 | 6.0e-7 | 1.2e-10 | 2e-15 |

At ratio ≤ 100, K = 8 is at 3.5e-13. **Default K = 8**: below Clarabel's tolerance for
any ratio the geodesic solver will see, at 8 power cones + 1 linear row per edge per
time step instead of 1 RSOC (roughly a 4–8× larger conic problem; see §5).

Two things make this clean rather than a hack:
- `Λ_K` is itself an admissible mean (symmetric because Gauss–Legendre nodes/weights
  are symmetric about ½; 1-homogeneous; concave as a positive combination of concave
  power means; continuous; positive). So the SOCP with `Λ_K` is *exact* for the metric
  built on `Λ_K`, not an approximation of anything. The shooting path can use either the
  exact `Λ` or the same `Λ_K`; using `Λ_K` in both keeps the two solvers solving the
  same problem, which is what the cross-validation gates need.
- The K-dependence is testable: `W2(Λ_K)` should converge to `W2(Λ)` at the rate of the
  table above.

---

## 3. Theory concerns, in order of how much they matter

1. **Boundary behaviour of the arithmetic mean.** For geometric, harmonic and
   logarithmic means `θ(0,t) = 0`: an empty node has zero mobility, so mass cannot leave
   it in a single step and geodesics between interior measures stay interior (this is
   the property the shooting positivity floor relies on). For the arithmetic mean
   `θ(0,t) = t/2 > 0`: the action `m²/θ` stays finite at an empty node, the SOCP will
   happily produce paths that touch `ρ = 0` (its `ρ ≥ 0` bound does the work), and the
   Hamiltonian flow can drive a density *through* zero in finite time. Consequences:
   `exp_map`/`log_map` with the arithmetic mean will hit the positivity floor more often
   and for legitimate reasons, so `:socp` must be the default recommendation for that
   mean, and the shooting tests need interior pairs chosen with more margin. This is
   not a bug to fix; it is the geometry. The paper should say which mean each figure
   uses.

2. **Which mean is "the" discrete transport metric.** Maas's result that the heat flow
   `ρ̇ = ρ Q` is the gradient flow of the relative entropy holds **only for the
   logarithmic mean**. The geometric mean (what the code has always used, following
   Erbar et al. 2020's computational choice) does not have that property. If any
   theory in the paper leans on entropy-gradient-flow structure or on the
   Erbar–Maas Ricci-curvature bounds, those statements are specific to `Λ`; the
   barycenter/analysis machinery (Fréchet means, Gram-matrix coordinate recovery,
   KKT stationarity) is mean-agnostic and holds for every admissible θ. Adding `Λ`
   therefore also lets the code compute the metric the theory papers are actually
   about, which may be worth a sentence in the write-up.

3. **Convexity is all the SOCP needs, but the cone forms must be exact.** Any error
   in a cone form is a wrong metric that converges beautifully (the geometric case's
   factor-of-2 history). Each new cone form gets the same two gates Module 1 had: the
   two-node closed form and the cross-check against shooting.

4. **Two-node closed form generalises.** With `ρ(r) = [1−r, 1+r]` and `π = [½,½]`,
   `W(ρ(s), ρ(t)) = (1/√2) ∫_s^t θ(1−r, 1+r)^{−1/2} dr` for any admissible θ (the
   geometric case gives the `(1−r²)^{−1/4}` integrand already in the tests). So the
   strongest gate in the suite is available for every mean for free, and it also
   pins the constant in front of each cone form.

5. **Shooting derivatives.** `φ̇` needs `∂₁θ`. Harmonic: `2t²/(s+t)²`, bounded (→2 as
   s→0), i.e. *better* behaved near the floor than geometric's `½√(t/s)`. Arithmetic:
   constant ½. Logarithmic: `logmean_partial_s` exists but switches to a series near
   `s = t` with a hard tolerance; for ForwardDiff to be exact through `log_map`'s
   Jacobian, both `Λ` and `∂₁Λ` need a smooth series branch near `s = t` (value and
   derivative consistent), or use `Λ_K`, which is smooth everywhere and needs no branch.

6. **Analysis is mean-agnostic but must be told the mean.** `potential_gram_qp`
   weights by `κθ(target)`; the SOCP's endpoint duals and the flow's `φ0` are still the
   gradient of `W²` and the velocity potential respectively, whatever θ is (the KKT
   argument in `SynthesisAnalysisMismatch.jl` never used the form of θ). Recovery is
   exact at the synthesis mean and N; analysing with a *different* mean is another
   convention mismatch, of the same kind as mismatched N, and should be shown once in
   an experiment rather than guarded against.

7. **The momentum ↔ potential identity** `m_t = −(1/2h) θ(ρ̄_t) ∇ψ_t` (from the
   action-epigraph RSOC's KKT conditions) holds for any mean, since the mean only enters
   through `ϑ`; the calibration test carries over with `θ` swapped.

---

## 4. Implementation plan (five small PRs)

### PR A: `src/core/Means.jl` (pure addition)
```julia
abstract type AdmissibleMean end
struct GeometricMean   <: AdmissibleMean end
struct ArithmeticMean  <: AdmissibleMean end
struct HarmonicMean    <: AdmissibleMean end
struct LogarithmicMean <: AdmissibleMean end            # exact Λ; series branch near s=t
struct QuadLogMean     <: AdmissibleMean; α::Vector{Float64}; w::Vector{Float64}; end
QuadLogMean(K::Int=8)                                    # Gauss–Legendre on [0,1] (Golub–Welsch, no dependency)

(θ::AdmissibleMean)(s, t)          # value, generic in eltype for ForwardDiff
partial_s(θ::AdmissibleMean, s, t) # ∂₁θ; partial_t by symmetry
```
`metric_tensor(G, ρ, θ::AdmissibleMean)` and `metric_tensor(ρ, θ::AdmissibleMean)`;
keep `geomean`/`logmean` functions as thin wrappers so nothing breaks. Tests:
symmetry, homogeneity, `θ(s,s) = s`, numerical concavity (midpoint inequality on a
grid), `∂₁θ` vs ForwardDiff, the ordering harmonic ≤ geometric ≤ Λ ≤ arithmetic,
`QuadLogMean(K)` vs `LogarithmicMean` with the table of §2.

### PR B: SOCP mean cones
`_geodesic_block!(model, G, N, h, left, right; mean=GeometricMean())` calls
`_mean_cone!(model, mean, ρ̄x, ρ̄y, ϑ)` with one method per type (linear / RSOC /
harmonic RSOC / K power cones + sum). `geodesic_socp`, `barycenter_socp`, `analyze_socp`
take `mean`. Gates: the generalised two-node closed form for each mean (O(h)
convergence), and `QuadLogMean(8)` vs `QuadLogMean(12)` agreeing to solver tolerance.

### PR C: shooting
`hamiltonian`, `hamiltonian_flow`, `integrate_hamiltonian`, `exp_map`, `log_map`,
`log_map_mollified`, `weighted_laplacian`, `momentum_to_potential`, `analyze_shooting`
take `mean` (default `GeometricMean()`), replacing the hard-coded `sqrt(ρ[x]*ρ[y])` and
`0.5*sqrt(ρ[y]/ρ[x])`. Gates: mass/H conservation per mean; `2H` vs `geodesic_socp.W2`
per mean (with the *same* mean object on both sides, so `QuadLogMean` on both);
two-node closed form per mean; `m0`/`W2` O(h) cross-check per mean; the `−2`
endpoint-potential relation per mean. Arithmetic-mean shooting tests use pairs with
extra interior margin (§3.1).

### PR D: analysis + unified API
`potential_gram_qp(G, target, potentials; mean)`; `mean=` keyword on `geodesic`,
`transport_cost`, `barycenter`, `analysis`, forwarded to `:socp`/`:shooting`, and an
`ArgumentError` for `:chambolle_pock` with any mean other than `GeometricMean()`
(`:sinkhorn` ignores it: its geometry is the ground cost). Round-trip test per mean.
Docs: a short "Admissible means" section in `api.md` with the table of §2 and the
boundary caveat of §3.1.

### PR E: experiments
`means_comparison.jl`: same references, four means: geodesic paths side by side,
barycenters, `W2` ordering check, wall-clock (power cones are slower), and the
cross-mean analysis mismatch (§3.6) shown once.

---

## 5. Cost and risk

- Arithmetic and harmonic: no cost change (one linear or RSOC row per edge-step,
  same as now). Shooting unchanged in cost.
- `QuadLogMean(8)`: 8 power cones + 1 row per edge per time step. Expect the SOCP to
  be several times slower and to scale worse with N and |E|; on MA house at N=10 the
  current 8 s could become tens of seconds. Shooting is unaffected (just a different
  θ), so **for the logarithmic mean, shooting is the practical solver** on interior
  data, and the SOCP is the certificate.
- `LogarithmicMean` (exact) in shooting with ForwardDiff: needs the smooth series
  branch; otherwise Newton's Jacobian is inconsistent near `ρ(x) ≈ ρ(y)` (a common
  situation on flat regions). Using `QuadLogMean` everywhere sidesteps this entirely
  and is the recommendation unless the exact `Λ` is needed for a theory comparison.
- Numerical: the harmonic mean's cone is exact; the power cones are well-conditioned
  as long as `ρ̄ > 0` (they are, by the positivity of interior data, and Clarabel
  handles the boundary of the cone itself).
