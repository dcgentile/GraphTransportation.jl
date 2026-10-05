# Folding Sinkhorn into the unified `method=` API — design

Question: should `barycenter(G, refs, λ; method=:sinkhorn)` exist, given that Sinkhorn works
on a ground-cost matrix rather than on the Markov chain?

Short answer: yes for `barycenter` (and later `analysis`), **but only with `cost` and `epsilon`
as required keywords and a docstring that says it is a different mathematical object.** Not for
`geodesic`/`transport_cost`. Details and a one-PR sketch below.

## 1. What the Sinkhorn code computes today

All in `src/sinkhorn/Sinkhorn.jl`, included at `src/GraphTransportation.jl:66`, exported at `:79`.

**`sinkhorn_differentiate(coords, measures, target, cost, epsilon, iters)`** (`Sinkhorn.jl:81-126`).
Bonneel–Peyré–Cuturi 2016 "Wasserstein barycentric coordinates": the iterative-Bregman
barycenter fixed point (Benamou et al. 2015) plus its hand-written reverse-mode derivative.
- `coords`: `λ ∈ Δ^p` (length `p`, sums to 1).
- `measures`: `n × p` matrix whose columns are **probability vectors** (`sum == 1`). Not
  densities w.r.t. `π`; the scripts feed `M_prob`, never `M_dens` (`examples/GridComparison.jl:91-92,195`).
- `target`: probability vector, or `nothing` (then no gradient is computed).
- `cost`: dense `n × n` ground cost `C`; only enters through `K = exp.(-C ./ ε)` (`:9-11, :87`).
- `epsilon`: entropic regularisation.
- `iters`: fixed iteration count; no convergence test. The loop runs `2:iters`, so `iters=1`
  returns the zero vector (confirmed).
- Returns `(p, w)`: `p = Πᵢ φᵢ^{λᵢ}` (`:98`) is the entropic barycenter, i.e. the minimiser of
  `Σᵢ λᵢ OT_ε(μᵢ, p)` for the ground cost `C`; `w` is `∂/∂λ ½‖p(λ) − target‖₂²` (`:105-123`).

**`sinkhorn_barycenter(coords, measures, target, cost, epsilon; iters=256)`** (`:135-138`, exported).
Returns `p` only; `target` is dead weight here (every caller passes `nothing`).

**`barycentric_loss` / `loss_gradient`** (`:148-154`, `:163-169`). Loss `½‖p(softmax(x)) − target‖²`
with `iters` hard-coded to 1024 and 2048 respectively. `logarithmic_change_of_variable` (`:19`) is a softmax.

**`build_geodesic(measures, cost; epsilon=0.1, steps=10, iters=2048)`** (`:180-191`, not exported,
untested). Two-point barycenters at weights `(1−t, t)`: an entropic displacement interpolation.
No momentum, no potentials, no action value.

**`simplex_regression(measures, target, cost, epsilon)`** (`:203-213`, exported).
L-BFGS over pre-softmax coordinates from the uniform start, returns `softmax(minimiser)`.
**Broken from the package namespace**: `Optim` is not in `Project.toml` and not imported in
`src/GraphTransportation.jl`; calling the export gives
`UndefVarError: Optim not defined in GraphTransportation` (confirmed by running it). Every
script sidesteps this with `using Optim; include("../src/sinkhorn/Sinkhorn.jl")` so the
re-included definitions shadow the module's (`examples/GridComparison.jl:25,32-35`,
`examples/EntropicComparison.jl:25-28`).

**Relation to `analysis`.** `analysis` (`src/API.jl:134-140`, QP in `src/core/Analysis.jl:16-42`)
solves `min_{λ∈Δ} λᵀAλ`, a stationarity test built from one geodesic per reference; it is a
convex QP with a closed-form Gram matrix. `simplex_regression` solves
`min_{λ∈Δ} ½‖bar_ε(λ) − target‖₂²`, a non-convex regression through the barycenter map, with
one forward+backward Sinkhorn per L-BFGS evaluation. Both return a simplex vector and both are
used as "coordinate recovery" in the comparison figures, but the objectives, the notion of
"recovered", and the cost profile are all different.

**How the scripts build `cost`** (`examples/GridComparison.jl:95-124`; identical in
`StatesComparison.jl`, `EntropicComparison.jl:93-99`):
- `:shortest_path`: unweighted hop distance via `Graphs.dijkstra_shortest_paths`
  (`src/experiments/ExperimentUtils.jl:73-85`), **squared**, then `./= maximum`.
- `:diffusion`: `D_t(i,j)² = Σ_k (Pᵗ[i,k] − Pᵗ[j,k])² / π[k]` with `P` the uniform random walk on
  the binary adjacency (`ExperimentUtils.jl:94-105`), `t = diameter`, squared, `./= maximum`.
  For the package's chains built by `markov_chain_from_edge_list` this `P` equals `G.Q`
  (row-stochastic, confirmed); for weighted chains it would not.
- Normalisation to `[0,1]` is load-bearing: without it `K` underflows at the ε values used
  (`EntropicComparison.jl:87-92`; also in the session memory). ε used: 0.0125 (grid), 0.125
  (states), 0.01 (entropic sweep). Nothing in the library enforces or documents this.

## 2. Options

The contract of the unified API is "same object, different algorithm" (`src/API.jl:1-3`,
`docs/src/api.md:11-12`). Sinkhorn violates it: the entropic barycenter for a chosen ground
metric is not the Erbar–Maas barycenter for any `ε` or any `cost`. Every option below has to
manage that.

### (a) Full integration with named cost rules and defaults
`barycenter(G, refs, λ; method=:sinkhorn, cost=:shortest_path, epsilon=0.01, iters=256)` with
`cost ∈ (:shortest_path, :diffusion, ::AbstractMatrix)` derived from `G` inside the library.
- `J`: `Σᵢ λᵢ ⟨C, Pᵢ⟩` where `Pᵢ` is the entropic plan between `μᵢ` and `ν` (or NaN).
  `info = (; cost, epsilon, iters, marginal_errors)`.
- `analysis(...; method=:sinkhorn)` → `simplex_regression` (needs Optim, see §4).
- `geodesic`: `build_geodesic` could fill `GeodesicSolution.ρ` with `m, φ0, φ1` NaN; `transport_cost`
  could return `sqrt(OT_ε)`. Both are misleading: `sqrt(OT_ε)` is not a metric, and the
  `ρ`-only path is not a geodesic of anything the rest of the package talks about.
- Misleads: `barycenter(G, refs, λ; method=:sinkhorn)` "just works" and silently picks a metric
  and an `ε`; its `J` sits next to the SOCP's `J` in a tuple of the same shape and invites
  comparison (cf. the cross-method optimality check at `test/runtests.jl:1064-1066`, which is
  meaningless for Sinkhorn). Also `barycenter(G, [μ], [1.0]; method=:sinkhorn)` ≠ `μ` (entropic
  blur), which no other method does.

### (b) Separate `entropic_*` family, outside `method=`
`entropic_barycenter(cost, refs, λ; epsilon, iters) -> (ν, J, info)`,
`entropic_coordinates(cost, refs, target; epsilon)`, plus `ground_cost(G, rule; t)` to build
`cost`. Documented as "a different object, provided for comparison".
- Cleanest statement of the mathematics; no `method=` contract bent.
- But: two vocabularies for the same experiment (`barycenter(G, …)` vs
  `entropic_barycenter(C, …)`), density/probability conversion left to the user, and it does
  nothing about `simplex_regression` being broken unless done anyway. It is essentially the
  status quo with better names.

### (c) Hybrid: `method=:sinkhorn` accepted, `cost` and `epsilon` mandatory
`barycenter(G, refs, λ; method=:sinkhorn, cost::AbstractMatrix, epsilon::Real, iters=256)`;
omitting `cost` or `epsilon` is an `ArgumentError`. A separate exported helper
`ground_cost(G, :shortest_path | :diffusion; t=diameter(G), normalize=true)` returns the squared,
`[0,1]`-normalised matrix the scripts build by hand, so the metric choice is a visible line in
user code rather than a default.
- `J`, `info` as in (a). `refs` are densities w.r.t. `G.π` like every other method; the wrapper
  converts `μᵢ = refsᵢ .* π` in and `ν = p ./ π` out, and the docstring says so.
- `analysis(...; method=:sinkhorn, cost, epsilon)` → `simplex_regression`, gated on Optim
  (§3, PR 2).
- `geodesic`/`transport_cost`: **not supported**; `test/runtests.jl:1046` already pins the
  `ArgumentError` and should stay.
- Misleads less than (a) because nothing is chosen for the user, and `J` is documented as
  `Σᵢ λᵢ ⟨C, Pᵢ⟩` for the entropic plans, not `Σᵢ λᵢ 𝒲²`.

## 3. Recommendation: (c)

Reasoning: the user's own framing is right on both counts. The consistency argument is about
*call sites* — the comparison experiments want to run one loop over methods on the same `G`
and `refs`, and today the Sinkhorn branch needs different input conventions (probabilities),
a hand-built cost, and a file `include` hack. The "fundamentally different" argument is about
*defaults and semantics*. (c) gives the call-site consistency while refusing to invent a
metric or an `ε` on the user's behalf, and the required keywords are the honest signal that
this branch is parameterised by things the other methods do not have. Add `:sinkhorn` to
`BARYCENTER_METHODS` (`src/API.jl:7`) only; leave `GEODESIC_METHODS` alone.

### PR 1 (small): `barycenter(...; method=:sinkhorn)` + `ground_cost`

`src/sinkhorn/Sinkhorn.jl` (additions, no changes to the existing functions):
```julia
ground_cost(G::MarkovGraph, rule::Symbol; t::Int=graph_diameter(G), normalize::Bool=true) -> Matrix{Float64}
    # :shortest_path — BFS hop counts over G.E (unweighted, matches compute_graph_metric), squared
    # :diffusion     — D_t² with P = Matrix(G.Q), π = G.π  (matches form_diffusion_map_from_graph
    #                  for unweighted chains; documented to use G.Q for weighted ones)
    # normalize      — ./= maximum, as every script does
_sinkhorn_plan(K, μ, ν; iters) -> Matrix   # 2-marginal Sinkhorn, ~10 lines; gives Pᵢ for J
```
BFS needs ~15 lines and avoids adding `Graphs` as a package dependency (it is only in
`examples/Project.toml`). Default `t = diameter` reproduces `GridComparison.jl:102`.

`src/API.jl`:
```julia
const BARYCENTER_METHODS = (:socp, :chambolle_pock, :sinkhorn)

function _barycenter_sinkhorn(G, refs, λ; cost=nothing, epsilon=nothing, iters::Int=256)
    cost    === nothing && throw(ArgumentError("barycenter(method=:sinkhorn) requires cost=  (see ground_cost)"))
    epsilon === nothing && throw(ArgumentError("barycenter(method=:sinkhorn) requires epsilon="))
    μ = reduce(hcat, (r .* G.π for r in refs))              # densities -> probability vectors
    p = sinkhorn_barycenter(λ, μ, nothing, cost, epsilon; iters)
    K = regularize_cost(cost, epsilon)
    plans = [_sinkhorn_plan(K, μ[:, i], p; iters) for i in eachindex(refs) if λ[i] > 0]
    J = sum(λ[i] * dot(cost, P) for (i, P) in zip(findall(>(0), λ), plans))
    marginal_errors = [norm(vec(sum(P, dims=2)) .- μ[:, i], 1) for (i, P) in zip(findall(>(0), λ), plans)]
    return p ./ G.π, J, (; cost, epsilon, iters, marginal_errors)
end
```
and one line in `barycenter` dispatching on `method == :sinkhorn`. Docstring paragraph (the
important part):

> `:sinkhorn`: the **entropically regularised Wasserstein barycenter for the ground cost `cost`**
> (Benamou et al. 2015 / Bonneel et al. 2016). This is not the discrete transport barycenter:
> it depends on the choice of `cost` (see `ground_cost`) and on `epsilon`, and even for a
> single reference it returns a blurred copy of that reference. `J = Σᵢ λᵢ ⟨cost, Pᵢ⟩` for the
> entropic plans `Pᵢ` (no entropy term), and is not comparable with the `J` of the other
> methods. Requires `cost` and `epsilon`; `refs` are densities w.r.t. `π` as elsewhere.

Docs: `docs/src/api.md:11-12` (mention `:sinkhorn` for `barycenter` only) and add `ground_cost`
to the `:55-60` section. `src/GraphTransportation.jl:79` exports `ground_cost`.

Tests (`test/runtests.jl`, new testset next to the unified-API one at `:1012`):
1. `ground_cost(G, :shortest_path)` on `grid_markov_chain(3)`: symmetric, zero diagonal,
   corner-to-corner `== 1.0`, adjacent `== 1/16`.
2. `ground_cost(G, :diffusion)`: symmetric, zero diagonal, `maximum == 1`, all finite.
3. **Equivalence pin**: `barycenter(G, refs, λ; method=:sinkhorn, cost=C, epsilon=ε)[1] .* π ≈
   sinkhorn_barycenter(λ, hcat((r .* π for r in refs)...), nothing, C, ε)` — the wrapper is
   exactly the old function under the density/probability conversion.
4. `dot(ν, π) ≈ 1`, `ν ≥ 0`, `J ≥ 0`, `all(marginal_errors .< 1e-6)`.
5. `ArgumentError` when `cost` or `epsilon` is omitted; `geodesic(...; method=:sinkhorn)` still
   throws (existing `:1046`).

Scripts: none need to change in PR 1. Migration of `examples/GridComparison.jl:95-124,195-204`
and `examples/StatesComparison.jl:153-162` to `ground_cost` + `barycenter(...; method=:sinkhorn)`
is PR 3, after `analysis` is done, because those scripts also call `simplex_regression`. The
`src/experiments/` copies are stale (see §4) and should not be touched.

### PR 2: `analysis(...; method=:sinkhorn)` via a package extension
Optim must come from somewhere. Options: (i) hard dependency (adds Optim + its tree to a package
whose users may never call it); (ii) `[weakdeps] Optim` + `ext/GraphTransportationOptimExt.jl`
defining `_analysis_sinkhorn` (Julia ≥ 1.9, which `Project.toml` already requires); (iii)
replace L-BFGS with an in-house projected gradient on the simplex (no dep, but changes the
numerics and inherits the gradient issue in §4). Recommend (ii): `analysis(G, target, refs;
method=:sinkhorn, cost, epsilon)` errors with "load Optim to use method=:sinkhorn" unless the
extension is active; the same conversion `.* π` as PR 1; returns `λ̂` only (`return_system` and
`compute_condition` have no analogue and should raise `ArgumentError`). Before this ships, the
gradient must be finite-difference-tested (§4 item 2).

## 4. Fragile or undocumented things a refactor has to face

1. **`simplex_regression` is unusable as exported** (`Sinkhorn.jl:211`; no Optim in
   `Project.toml`). The docstring says "Requires Optim.jl (`using Optim` must be available in
   the calling scope)", which is not how Julia scoping works — it only works because the scripts
   re-`include` the file into `Main`. PR 2 must resolve this one way or another.
2. **The gradient handed to L-BFGS is not the gradient of the objective.** `barycentric_loss`
   is a function of pre-softmax `x` (`:148-154`), but `loss_gradient` returns `w`, which
   `sinkhorn_differentiate` computes as `∂/∂λ` at `λ = softmax(x)` (`:105-123`, `:163-169`);
   the softmax Jacobian is never applied (`:207-210`). This is a plausible cause of the
   "unreliable simplex_regression gradients" and try/catch around it in
   `examples/EntropicComparison.jl:103,123-139,168-174`. Needs a finite-difference test before
   `analysis(method=:sinkhorn)` exists; the fix is a one-line chain rule in `g!`.
3. **Iteration counts are hard-coded and inconsistent**: 256 (`sinkhorn_barycenter`), 1024
   (`barycentric_loss`), 2048 (`loss_gradient`). No convergence criterion anywhere; the wrapper's
   `marginal_errors` in `info` is the first diagnostic a user would get.
4. **Memory**: `b` and `phi` are `n × p × iters` (`:83,:86`); at `iters=2048` this is ~8 MB for
   the 160-node MA graph but ~250 MB per array at `n = 5000`. Fine for the current graphs; note it.
5. **Cost normalisation and kernel underflow** are the user's responsibility today
   (`EntropicComparison.jl:87-92`, `kernel_stats` in the scripts). There is no log-domain
   stabilisation, so small `ε` on an unnormalised cost silently produces NaN/zeros.
   `ground_cost(...; normalize=true)` puts the convention in one documented place.
6. **Input convention mismatch**: Sinkhorn takes probability vectors, the unified API takes
   densities w.r.t. `π`. For non-uniform `π` (any non-regular graph) the raw call and the
   wrapped call differ unless the conversion is applied; PR 1's equivalence test pins this.
7. **Dead / unused code**: `target` argument of `sinkhorn_barycenter` (`:135`); `sqeuc_loss_grad`,
   `ell_one_loss`, `kl_loss` (`:31-53`) have no callers; `build_geodesic` (`:180`) is unexported
   and untested.
8. **No tests at all for `src/sinkhorn/`**; the only mention in `test/runtests.jl` is the
   `method=:sinkhorn` throw at `:1046`.
9. **Stale experiment copies**: `src/experiments/GridComparison.jl:36` includes `"../Sinkhorn.jl"`,
   a path that no longer exists after the src-layout move in PR #29; `examples/` has the live
   versions (`examples/GridComparison.jl:35` is correct). Any migration should touch only `examples/`.
10. **Diffusion cost provenance**: the scripts build `P` from the binary adjacency, which equals
    `G.Q` only for uniform random walks. `ground_cost(G, :diffusion)` using `G.Q` is the natural
    generalisation but is a (documented) change for weighted chains such as
    `weighted_hypercube_markov_chain`.
