# Five Routes to Fast Computation of Discrete Transport Geodesics, Distances, and Barycenters on Graphs

*Prepared as a technical companion to the barycentric coding model of Gentile & Murphy, building on the discretization of Erbar, Rumpf, Schmitzer & Simon [10]. Notation follows [10] and [24] throughout.*

---

## 0. Setting and the computational bottleneck

Let $(\mathcal{X}, Q, \pi)$ be a finite, irreducible, reversible Markov triple, $\mathcal{P}(\mathcal{X})$ the densities w.r.t. $\pi$, and $\theta$ an admissible mean. The discrete transport distance of Maas [24] and Chow–Huang–Li–Zhou [6] is

$$\mathcal{W}(\rho_A,\rho_B)^2 = \inf\Big\{ \mathcal{A}(\rho,m) := \tfrac12 \int_0^1 \sum_{x,y} \frac{m_t(x,y)^2}{\theta(\rho_t(x),\rho_t(y))}\, Q(x,y)\pi(x)\, dt \;:\; (\rho,m)\in \mathcal{CE}(\rho_A,\rho_B)\Big\},$$

a graph analogue of Benamou–Brenier [3]. The reference numerical scheme [10] performs a Galerkin discretization in time (piecewise-affine $\rho$, piecewise-constant $m$, $N$ intervals, $h=1/N$), proves $\Gamma$-convergence as $h\to 0$, and solves the resulting finite-dimensional convex program with Chambolle–Pock (CP) primal–dual splitting [5] after introducing slack variables to decouple the mean nonlinearity.

Three properties of this pipeline dominate its cost. First, CP is a first-order method with an $O(1/k)$ ergodic rate on this problem class, and the tail is slow: reaching the tolerances used in practice ($10^{-10}$ in [10] and in our own experiments) requires $10^4$–$10^5$ iterations. Second, every iteration performs a global space–time elliptic solve (the projection onto the discrete continuity equation, §4.2 of [10]). Third, the variable count is $O(N\cdot|E|)$ with $N$ in the tens to thousands. The five strategies below attack, respectively: the algorithm class (§1, §2), the dimensionality of the problem actually solved (§3), the necessity of solving it at all (§4), and the constants of the existing scheme (§5).

Throughout, $\kappa_e := Q(x,y)\pi(x)$ for the undirected edge $e=(x,y)$ (well defined by reversibility), $\langle\varphi,\psi\rangle_\pi := \sum_x \varphi\psi\,\pi$, $\langle\Phi,\Psi\rangle_Q := \tfrac12\sum_{x,y}\Phi\Psi\,Q\pi$, $\nabla\varphi(x,y)=\varphi(x)-\varphi(y)$, and $\mathrm{div}$ is the negative adjoint of $\nabla$ with respect to these inner products, so that $\langle \varphi, \mathrm{div}\,\Phi\rangle_\pi = -\langle \nabla\varphi, \Phi\rangle_Q$.

---

## 1. Exact conic reformulation and interior-point methods

**Claim.** For $\theta = \theta_{\mathrm{geo}}$, the time-discretized problem $\mathcal{W}_h^2$ of [10] — and the joint barycenter problem $\min_\nu \sum_i \lambda_i \mathcal{W}_h^2(\nu_i,\nu)$ — are *exactly* second-order cone programs (SOCPs). No smoothing, regularization, or approximation is introduced; only the algorithm changes.

**Construction.** Lemma 4.1 of [10] already performs the essential epigraph step: for each edge–interval pair,

$$\alpha(s,t,m) = \inf\{\, \Lambda(\vartheta, m) : 0 \le \vartheta \le \theta(s,t) \,\}, \qquad \Lambda(\vartheta,m)=\tfrac{m^2}{\vartheta},$$

by monotonicity of $\Lambda$ in $\vartheta$. Introduce a further epigraph variable $w$ for $\Lambda$. Two three-dimensional convex sets remain:

$$\mathcal{K}_1 = \{(m,\vartheta,w): m^2 \le \vartheta w,\ \vartheta,w\ge 0\}, \qquad \mathcal{K}_2 = \{(s,t,\vartheta): 0\le \vartheta,\ \vartheta^2 \le st,\ s,t \ge 0\}.$$

Both are rotated second-order cones, via the polarization identity $4u^2 \le ab \iff \|(2u,\,a-b)\|_2 \le a+b$ for $a,b \ge 0$; SOC-representability in the sense of [2, 22] follows since each is an affine preimage of the Lorentz cone. $\mathcal{K}_1$ encodes the Benamou–Brenier integrand exactly, including its lower-semicontinuous closure at $\vartheta = 0$ (the set is the closed epigraph of the perspective function of $m\mapsto m^2$; cf. the dual description $\Lambda^* = \iota_{\{p + q^2/4 \le 0\}}$ computed in §4.3 of [10] — the parabola there is precisely the polar of this cone). $\mathcal{K}_2$ is the paper's constraint set $K$ specialized to the geometric mean, with no relaxation: feasibility of the cone forces $s,t\ge 0$ automatically.

The full program has variables $\rho \in \mathbb{R}^{n\times(N+1)}$ and $(m,\vartheta,w) \in \mathbb{R}^{|E|\times N}$ each; linear equalities (continuity equation, time-averaging map $\bar\rho_i = (\rho_i+\rho_{i+1})/2$, boundary data); the cone memberships $(m,\vartheta,w)_{e,i}\in\mathcal{K}_1$, $(\bar\rho_x,\bar\rho_y,\vartheta)_{e,i}\in\mathcal{K}_2$; and linear objective $h\sum_{i,e}\kappa_e\, w_{e,i}$, whose optimal value is $\mathcal{W}_h(\rho_A,\rho_B)^2$. The barycenter version replicates the block $p$ times with fixed left endpoints $\nu_i$ and a shared free right endpoint $\nu \ge 0$, $\langle \nu, 1\rangle_\pi = 1$; convexity of the joint program is inherited immediately (each $\mathcal{W}_h^2(\nu_i,\cdot)$ is a partial minimization of a jointly convex function over a set in which $\nu$ enters affinely), so the synthesis problem is solved *globally, in one shot, with a duality-gap certificate*, eliminating the intrinsic gradient descent loop entirely — along with its step-size, tolerance, injectivity-radius, and geodesic-convexity concerns.

**Why this is fast.** Primal–dual interior-point methods for SOCPs [1, 22, 26] converge in $O(\sqrt{\nu_{\mathrm{bar}}}\log(1/\varepsilon))$ Newton iterations — in practice 25–50 iterations essentially independently of the target accuracy — each iteration being a sparse symmetric factorization of a KKT system whose sparsity is block-tridiagonal in time with graph-Laplacian spatial blocks. This replaces $10^4$–$10^5$ CP iterations with $\sim 10^{1.5}$ Newton steps. Off-the-shelf solvers (Mosek; Clarabel [16], which is native Julia) consume the model directly through JuMP.

**Relation to prior work, and the case of the logarithmic mean.** The observation that a judicious mean makes the graph Benamou–Brenier problem conic is due to Solomon et al. [30], who selected the *harmonic* mean precisely because $m^2/\theta_{\mathrm{har}}(s,t) = \tfrac12(m^2/s + m^2/t)$ splits into two rotated cones; [10] note that this "does not extend to other choices of $\theta$." For exact representation of the logarithmic mean the obstruction is fundamental: SOC-representable sets are semialgebraic (projections of basic semialgebraic sets, by Tarski–Seidenberg [4]), while the slice $t=1$ of $\{\vartheta \le \theta_{\log}(s,t)\}$ has boundary $\vartheta = (s-1)/\log s$, which is transcendental; hence $K_{\log}$ admits no exact SOC (indeed no semidefinite) representation. Two exits exist if the log mean is ever needed: (i) the integral representation $\theta_{\log}(s,t) = \int_0^1 s^{1-\lambda}t^{\lambda}\, d\lambda$ (Carlson [7]) together with a quadrature $\sum_j w_j s^{1-\lambda_j} t^{\lambda_j}$, each term a three-dimensional *power cone* constraint (supported by Mosek and Clarabel [16, 25]); convexity of $\lambda \mapsto s^{1-\lambda}t^\lambda$ makes midpoint-type rules *under*-estimate $\theta_{\log}$, yielding an inner approximation of $K$ and hence a certified upper bound on $\mathcal{W}$, with a trapezoid counterpart bounding from below; or (ii) the methods of §2–§3, which never require conic structure. For the barycentric coding model as implemented (geometric mean), none of this is needed and the representation is exact.

---

## 2. Regularize-and-Newton

If one prefers to keep an arbitrary admissible mean and the original (non-conic) formulation, the obstacle to second-order optimization is that the action is nonsmooth and degenerate exactly where mass vanishes: $\alpha$ is only lower semicontinuous at $\theta = 0$, and its Hessian blows up as densities approach the boundary of the simplex.

Li, Yin & Osher [21] showed, for dynamical optimal transport on grids, that augmenting the Benamou–Brenier action with a *Fisher information* term,

$$\mathcal{A}_\beta(\rho,m) = \mathcal{A}(\rho,m) + \beta\, \mathcal{I}(\rho), \qquad \mathcal{I}(\rho) \sim \int_0^1 \big\|\nabla\sqrt{\rho_t}\big\|^2 dt \ \text{(discretized)},$$

renders the problem smooth and strictly convex with minimizers uniformly in the interior, so that a damped Newton method converges quadratically; the regularization is not ad hoc but corresponds to a Schrödinger-bridge / entropic interpolation structure (Léonard [20]; Chen–Georgiou–Pavon [8]), and $\beta \to 0$ recovers the unregularized distance, enabling a continuation (path-following) strategy. The transcription to the graph setting is direct: add $\beta \int_0^1 \langle \nabla\sqrt{\rho}, \theta$-weighted $\nabla\sqrt{\rho}\rangle_Q\,dt$ (or the simpler unweighted graph Dirichlet energy of $\sqrt\rho$) to the time-discretized action of [10]; the KKT system becomes smooth, its Newton systems carry the same space–time sparsity as §1, and one factorizes with sparse Cholesky or preconditioned CG.

Two remarks temper the recommendation. First, the regularization introduces bias of the same conceptual flavor as entropic regularization of static OT — precisely the bias the dynamic approach of the barycentric coding model is advertised as avoiding — so for the paper's purposes §1 (fast *and* unbiased) dominates where it applies. Second, an unbiased second-order alternative exists in the form of semismooth Newton methods applied to the unregularized optimality system [17], at the cost of substantially more delicate implementation. We regard §2 primarily as the fallback of choice for means outside the conic-representable family, and as a theoretically interesting bridge to a discrete Schrödinger problem (an answer, arguably, to Discussion item 4 of the barycenter paper).

---

## 3. Hamiltonian shooting for the geodesic two-point boundary value problem

The variational problem defining $\mathcal{W}$ is a geodesic problem on a (formal) Riemannian manifold [24, 6, 23], and geodesics satisfy a system of ODEs of dimension $2n$, $n = |\mathcal{X}|$ — independent of any time discretization. When applicable, solving this ODE boundary value problem by shooting is dramatically cheaper than the space–time convex program, and it furnishes the exponential and logarithm maps that the barycentric coding model consumes directly. We derive the system carefully, then discuss the numerics and the domain of validity.

### 3.1 Derivation of the geodesic equations

Work formally in the interior of the simplex (all densities strictly positive); rigorous treatments are [24, §3], [11], and the Hamilton–Jacobi analyses [14, 15]. Write the action as

$$\mathcal{A}(\rho,m) = \int_0^1 \big\langle m_t \oslash \theta(\rho_t),\, m_t \big\rangle_Q\, dt,$$

where $\oslash$ is entrywise division and $\theta(\rho)(x,y) := \theta(\rho(x),\rho(y))$; the factor $\tfrac12$ of the original expression is absorbed by the $\tfrac12$ in $\langle\cdot,\cdot\rangle_Q$. We minimize over pairs subject to the continuity equation $\partial_t\rho + \mathrm{div}\, m = 0$ with $\rho_0 = \rho_A$, $\rho_1 = \rho_B$. Introduce a time-dependent Lagrange multiplier $\varphi_t \in \mathbb{R}^{\mathcal{X}}$ and form

$$\mathcal{L}(\rho, m, \varphi) = \int_0^1 \big\langle m\oslash\theta(\rho),\, m\big\rangle_Q\, dt + \int_0^1 \big\langle \varphi,\ \partial_t \rho + \mathrm{div}\, m\big\rangle_\pi\, dt.$$

**Stationarity in $m$.** For a variation $\delta m$, using $\langle\varphi, \mathrm{div}\,\delta m\rangle_\pi = -\langle\nabla\varphi, \delta m\rangle_Q$:

$$\delta_m \mathcal{L} = \int_0^1 \big\langle \delta m,\ 2\, m\oslash\theta(\rho) - \nabla\varphi \big\rangle_Q\, dt = 0 \quad\Longrightarrow\quad m_t = \tfrac12\, \theta(\rho_t)\circ \nabla\varphi_t.$$

Setting $\psi := \varphi/2$ (a harmless rescaling of the multiplier) gives the momentum in *gradient form*:

$$\boxed{\ m_t = \theta(\rho_t)\circ\nabla\psi_t\ }$$

— the graph analogue of Benamou–Brenier's $v = \nabla\phi$, and the statement that geodesic velocities lie in the tangent space $T_\rho = \{\nabla\psi\}$ identified in [24] and used to define the Gram matrix of the analysis problem.

**Stationarity in $\rho$.** The action depends on $\rho$ only through $\theta$. Differentiating $\tfrac12\sum_{a,b} m(a,b)^2\,\theta(\rho(a),\rho(b))^{-1} Q(a,b)\pi(a)$ with respect to $\rho(x)$ produces two sums (from $a=x$ and $b=x$) which coincide by the symmetry of $\theta$, the antisymmetry of optimal $m$ (Lemma 2.4 of [10]), and reversibility, giving

$$\partial_\rho \mathcal{A}\,(x) = -\sum_y \frac{m(x,y)^2}{\theta(\rho(x),\rho(y))^2}\ \partial_1\theta\big(\rho(x),\rho(y)\big)\, Q(x,y)\,\pi(x).$$

Integrating the constraint term by parts in time (boundary variations vanish since the endpoints are fixed), $\delta_\rho \int \langle\varphi,\partial_t\rho\rangle_\pi\,dt = -\int \langle \partial_t\varphi, \delta\rho\rangle_\pi\, dt$, so stationarity in $\rho(x)$ reads $\partial_t\varphi(x) = -\sum_y (m^2/\theta^2)\,\partial_1\theta\, Q(x,y)$. Substituting $m = \theta\nabla\psi$ (so $m^2/\theta^2 = (\nabla\psi)^2$) and $\varphi = 2\psi$:

$$\boxed{\ \partial_t \psi_t(x) = -\tfrac12 \sum_y \partial_1\theta\big(\rho_t(x),\rho_t(y)\big)\, \big(\nabla\psi_t(x,y)\big)^2\, Q(x,y)\ }$$

coupled with the continuity equation under the gradient ansatz,

$$\boxed{\ \partial_t \rho_t(x) = -\,\mathrm{div}\big(\theta(\rho_t)\circ\nabla\psi_t\big)(x) = \sum_y \theta\big(\rho_t(x),\rho_t(y)\big)\big(\psi_t(x)-\psi_t(y)\big)Q(x,y).\ }$$

The $\psi$-equation is a *discrete Hamilton–Jacobi equation*. Note the structural difference from the Euclidean case: there $\partial_t\phi + \tfrac12|\nabla\phi|^2 = 0$ decouples from $\rho$ (because the action density $\rho|v|^2$ is linear in $\rho$, so its $\rho$-derivative is $|v|^2$, independent of $\rho$), whereas on graphs $\partial_1\theta$ genuinely depends on $\rho$ — mass and phase remain coupled. This coupling is the analytic heart of the non-locality and boundary phenomena studied in [11, 14, 15].

**Hamiltonian structure and first integrals.** Define

$$H(\rho,\psi) := \tfrac12\big\langle \nabla\psi,\ \theta(\rho)\circ\nabla\psi\big\rangle_Q = \tfrac14\sum_{x,y}\theta\big(\rho(x),\rho(y)\big)\big(\psi(x)-\psi(y)\big)^2 Q(x,y)\pi(x).$$

A direct computation (using symmetry and reversibility exactly as above) verifies that the two boxed equations are Hamilton's equations for $H$ in the $\pi$-weighted pairing: $\pi(x)\,\partial_t\rho(x) = \partial H/\partial\psi(x)$ and $\pi(x)\,\partial_t\psi(x) = -\partial H/\partial\rho(x)$. Consequences: (i) $H$ is conserved along geodesics; (ii) total mass $\langle \rho, 1\rangle_\pi$ is conserved (it is the momentum map of the gauge symmetry $\psi \mapsto \psi + c$, under which $H$ is invariant — the discrete Noether pairing); (iii) the metric speed satisfies $\|\dot\rho_t\|_{\mathcal{W}}^2 = \langle\nabla\psi_t, \theta(\rho_t)\nabla\psi_t\rangle_Q = 2H$, constant, whence

$$\mathcal{W}(\rho_A,\rho_B)^2 = \mathcal{A}(\rho, \theta\nabla\psi) = \int_0^1 2H\, dt = 2\,H(\rho_0,\psi_0).$$

So the squared distance is read off the *initial data alone*. In Riemannian language, $H$ is (half) the co-metric: the metric tensor of [24] is $g_\rho = L_\theta(\rho)^{-1}$ acting on tangent vectors $\sigma = -\mathrm{div}(\theta(\rho)\nabla\psi)$, where $L_\theta(\rho)\psi := -\mathrm{div}(\theta(\rho)\circ\nabla\psi)$ is the $\rho$-weighted graph Laplacian; the geodesic flow above is the cogeodesic flow of this tensor (cf. Li's transport information geometry [23], which computes the associated Christoffel symbols and curvature explicitly).

For $\theta_{\mathrm{geo}}$ specifically, $\partial_1\theta_{\mathrm{geo}}(s,t) = \tfrac12\sqrt{t/s}$, so the $\psi$-equation is explicit and smooth for $\rho > 0$.

### 3.2 Shooting

The log map $\log_{\rho_A}(\rho_B)$ is the initial datum of the geodesic BVP: find $\psi_0$ (modulo constants — fix the gauge by $\langle\psi_0, 1\rangle_\pi = 0$) such that the flow launched from $(\rho_A, \psi_0)$ satisfies $\rho_1 = \rho_B$. This is a classical two-point BVP [31, Ch. 7], [1a := Ascher–Mattheij–Russell [29]]:

*Single shooting.* Define $F(\psi_0) := \rho_1(\rho_A, \psi_0) - \rho_B$ on the mean-zero subspace ($\dim n-1$; note $\langle F, 1\rangle_\pi = 0$ automatically by mass conservation, so the system is square). Newton's method on $F$ requires the sensitivity $\partial \rho_1/\partial \psi_0$ — the solution of the variational (linearized Hamiltonian) equations along the trajectory, obtainable by forward-mode automatic differentiation through the integrator or by finite differences; each Newton step is one ODE integration plus one dense $(n-1)\times(n-1)$ solve. Convergence is locally quadratic; in practice 3–8 iterations cold, 1–2 warm-started, since in any outer loop (analysis of a slowly varying target; descent-based synthesis if retained for comparison) the previous $\psi_0$ is an excellent initial guess.

*Initialization.* The linearization of the exponential map at $\rho_A$ is exactly the weighted-Laplacian solve: to first order in $\|\rho_B - \rho_A\|$, $L_\theta(\rho_A)\,\psi_0 = \rho_B - \rho_A$ (in the $\pi$-weighted sense, on mean-zero functions). One sparse SPD solve therefore provides an initializer that is asymptotically exact for nearby measures; alternatively, a coarse ($N=5$) run of the §1 SOCP supplies $\psi$ via the continuity-equation duals.

*Integration.* The system is Hamiltonian; Störmer–Verlet or another symplectic scheme [18] preserves $H$ and mass to high accuracy over $[0,1]$ and gives well-conditioned sensitivities, though at these horizons a small-step RK4 is also acceptable. *Multiple shooting* (partition $[0,1]$, impose matching conditions) is the standard remedy if single shooting exhibits sensitivity on stiff instances [29].

*Exponential map.* $\exp_\nu(t\,\sigma)$ is the same flow run forward: recover $\psi_0$ from a tangent vector or momentum by one $L_\theta(\nu)$ solve, integrate for time $t$. This replaces the first-order continuity-equation heuristic $\nu \mapsto \nu + \varepsilon\,\mathrm{div}(\sum_i\lambda_i m_i(0))$ with a genuine retraction, positivity-preserving for steps that remain in the interior.

### 3.3 Cost and domain of validity

Per log-map evaluation: a handful of ODE integrations of dimension $2n$ and dense solves of dimension $n-1$ — no time-discretized program, no $N$, no splitting tolerance. Against the $O(N|E|)$-variable convex program this is the asymptotically cheapest option by a wide margin, and the analysis Gram matrix $A_{ij} = \langle\nabla\psi_i, \theta(\nu)\circ\nabla\psi_j\rangle_Q$ is assembled exactly from the recovered potentials.

The method is, however, an *interior* method in an essential sense. $\partial_1\theta_{\mathrm{geo}}(s,t) = \tfrac12\sqrt{t/s} \to \infty$ as $s\to 0$: the Hamiltonian is not $C^1$ up to the boundary of the simplex, and this reflects genuine geometry rather than numerical fragility — geodesics can reach the boundary in finite time (with density vanishing quadratically, matching the $t^2$ rates in the constructions of [10, Prop. 3.3]), can do so even between interior endpoints [14], and uniqueness can fail there. The rigorous framework for the boundary is the Hamilton–Jacobi theory on the Wasserstein space over graphs [14, 15]. Practically: guard the flow with a positivity floor; for boundary-supported data either mollify, $\nu_\varepsilon = (1-\varepsilon)\nu + \varepsilon\mathbb{1}$, and extrapolate in $\sqrt{\varepsilon}$ (moving mass $\varepsilon$ costs $O(\sqrt\varepsilon)$ in $\mathcal{W}$, by the elementary-flow constructions of [10, §3.2]), or — better — route those instances to §1, which is entirely insensitive to boundary support.

---

## 4. Avoiding geodesics: linearization of the metric

Both halves of the barycentric coding model consume geodesics only through their initial data, and in several regimes even that is more than necessary.

**Local distances in one linear solve.** Since $g_\rho = L_\theta(\rho)^{-1}$ on the interior, for nearby measures

$$\mathcal{W}(\rho,\ \rho + \delta\sigma)^2 = \delta^2\, \big\langle \sigma,\ L_\theta(\rho)^{\dagger}\sigma\big\rangle_\pi + O(\delta^3), \qquad \langle\sigma,1\rangle_\pi = 0,$$

a single sparse SPD solve per query (factor $L_\theta(\rho)$ once per base point). Chained along a coarse path this yields convergent distance estimates with no optimization whatsoever, and it is the natural inner product for any local statistical analysis of measure-valued graph signals.

**Linearized coding model.** Freezing a base measure $\nu$ and replacing $\log_\nu(\nu_i)$ by the solution of $L_\theta(\nu)\psi_i = \nu_i - \nu$ turns the analysis Gram matrix into $A_{ij} = \langle \psi_i,\ \nu_j - \nu\rangle_\pi$ — assembly cost $p$ linear solves — and is precisely the graph counterpart of linearized optimal transport and the linearized BCM studied on Euclidean domains [33, 27, and the LBCM literature the barycenter paper already cites]. The leading-order barycenter under this linearization degenerates to the Euclidean average $\sum_i\lambda_i\nu_i$, which delimits its use: it is a cheap initializer, an error model, and a fast approximate analysis for targets near the references, not a replacement for the exact model. Quantifying the gap between the linearized and exact coordinates on graphs is, we think, a publishable question in its own right, given the Euclidean-case precedents [27, 33].

**Gradient-flow shortcut.** For JKO / minimizing-movement schemes [19, 10 §6], each step needs $\mathcal{W}^2(\rho_k,\cdot)$ only to the order of the outer time step; substituting the quadratic form above preserves first-order consistency of the flow while reducing each step to a smooth strictly convex problem in $n$ variables. This is peripheral to the coding model but relevant wherever the [10] JKO machinery is reused.

---

## 5. Accelerating the first-order splitting itself

If the CP scheme of [10] is retained anywhere (e.g., as a legacy baseline), substantial constant-factor gains are available without changing the mathematics.

**Tolerance matching.** The $\Gamma$-convergence theory gives an $O(h)$ discretization error (confirmed empirically in [10, Fig. 9]); solving the inner problem to $10^{-10}$ when $h = 10^{-1}$–$10^{-2}$ wastes the long tail of an $O(1/k)$ method. Our own hyperparameter study (recovery error flat down to $\delta_g \approx 10^{-8}$ and coarse $N$) says the same. Terminating at a tolerance proportional to the discretization error is the single cheapest speedup in this family.

**Algorithmic upgrades.** Diagonal preconditioning of PDHG [28]; adaptive step-size balancing [13]; and, most consequentially in recent practice, *restarted* PDHG — averaging/Halpern restarts convert the sublinear worst case into robust practical linear convergence and underlie the modern LP solver PDLP [1b := Applegate et al. [2a]]. Warm starts across outer iterations (Discussion item 9 of the barycenter paper) are correct and effective for first-order methods. Prefactoring the constant space–time elliptic matrix (already suggested for one projection in [10]) reduces each CP iteration's solve to two triangular sweeps.

**Hardware.** The conic form of §1 is what makes the problem GPU-native. A primal–dual iteration on the conic formulation consists of sparse matrix–vector products with the space–time incidence matrix plus millions of *independent three-dimensional* cone projections — no linear solves at all (the per-iteration elliptic solve of [10] is an artifact of that particular splitting, not of PDHG). This is exactly the workload behind the current generation of GPU conic solvers: cuPDLP for LP [2b], PDCS/cuPDCS for conic programs including SOCPs [3a], and the GPU interior-point solver CuClarabel [9a]. The empirical crossover reported in that literature matches this problem's profile: below $\sim 10^5$ variables (all experiments in the barycenter paper), CPU interior point wins and per-instance GPU use is dominated by transfer overhead — but *batching* hundreds of independent small instances (Monte-Carlo consistency experiments, per-analysis geodesic fans, $\lambda$-grids over the simplex) on a GPU yields one to two orders of magnitude of throughput; above $\sim 10^6$ variables (mesh-scale graphs, e.g. the 6094-vertex hand mesh of [10] at $N = 33$), GPU first-order conic solvers at $10^{-5}$-ish accuracy become the dominant tool.

---

## 6. Recommendations

For the barycentric coding model as constituted (geometric mean, graphs up to a few hundred nodes): implement §1 as the workhorse — joint SOCP synthesis, SOCP geodesics for analysis — because it is exact, certified, boundary-robust, and one-to-two orders of magnitude faster than splitting at the tolerances that matter; implement §3 as the fast path for interior data and as the principled exponential map, upgrading the synthesis-by-descent scheme from heuristic to bona fide Riemannian optimization where it is kept for exposition; use §4 for initializers, large-scale screening, and as the basis of a linearized BCM variant; apply §5's tolerance matching and batching to whatever first-order code survives. §2 is the designated fallback should the model ever move to the logarithmic mean (e.g., to align with the entropy gradient-flow theory of [24, 12]), where §1 is only approximately available and §3 remains exact but interior-only.

---

## References

[1] F. Alizadeh, D. Goldfarb. Second-order cone programming. *Math. Program.* 95 (2003) 3–51.

[1a] U. Ascher, R. Mattheij, R. Russell. *Numerical Solution of Boundary Value Problems for ODEs.* SIAM, 1995. (= [29])

[1b]/[2a] D. Applegate, M. Díaz, O. Hinder, H. Lu, M. Lubin, B. O'Donoghue, W. Schudy. Practical large-scale linear programming using primal–dual hybrid gradient (PDLP). *NeurIPS* 2021.

[2] A. Ben-Tal, A. Nemirovski. *Lectures on Modern Convex Optimization.* SIAM, 2001.

[2b] H. Lu, J. Yang. cuPDLP.jl: A GPU implementation of restarted PDHG for LP. arXiv:2311.12180, 2023.

[3] J.-D. Benamou, Y. Brenier. A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem. *Numer. Math.* 84 (2000) 375–393.

[3a] Z. Lin, Z. Xiong, D. Ge, Y. Ye. PDCS: A primal–dual large-scale conic programming solver with GPU enhancements. arXiv:2505.00311, 2025.

[4] J. Bochnak, M. Coste, M.-F. Roy. *Real Algebraic Geometry.* Springer, 1998. (Tarski–Seidenberg.)

[5] A. Chambolle, T. Pock. A first-order primal–dual algorithm for convex problems with applications to imaging. *J. Math. Imaging Vis.* 40 (2011) 120–145.

[6] S.-N. Chow, W. Huang, Y. Li, H. Zhou. Fokker–Planck equations for a free energy functional or Markov process on a graph. *Arch. Ration. Mech. Anal.* 203 (2012) 969–1008.

[7] B. C. Carlson. The logarithmic mean. *Amer. Math. Monthly* 79 (1972) 615–618.

[8] Y. Chen, T. T. Georgiou, M. Pavon. Stochastic control liaisons: Richard Sinkhorn meets Gaspard Monge on a Schrödinger bridge. *SIAM Rev.* 63 (2021) 249–313.

[9] M. Erbar, J. Maas. Ricci curvature of finite Markov chains via convexity of the entropy. *Arch. Ration. Mech. Anal.* 206 (2012) 997–1038.

[9a] Y. Chen, D. Tse, P. Nobel, P. Goulart, S. Boyd. CuClarabel: GPU acceleration for a conic optimization solver. arXiv:2412.19027; *ACM TOMS*, 2025.

[10] M. Erbar, M. Rumpf, B. Schmitzer, S. Simon. Computation of optimal transport on discrete metric measure spaces. *Numer. Math.* 144 (2020) 157–200.

[11] M. Erbar, J. Maas, M. Wirth. On the geometry of geodesics in discrete optimal transport. *Calc. Var. PDE* 58 (2019) 19.

[12] M. Erbar, J. Maas. Gradient flow structures for discrete porous medium equations. *Discrete Contin. Dyn. Syst.* 34 (2014) 1355–1374.

[13] T. Goldstein, M. Li, X. Yuan. Adaptive primal–dual splitting methods for statistical learning and image processing. *NeurIPS* 2015; see also arXiv:1305.0546.

[14] W. Gangbo, W. Li, C. Mou. Geodesics of minimal length in the set of probability measures on graphs. *ESAIM: COCV* 25 (2019) 78.

[15] W. Gangbo, C. Mou, A. Święch. Well-posedness for Hamilton–Jacobi equations on the Wasserstein space on graphs. *Calc. Var. PDE* 63 (2024) 160.

[16] P. Goulart, Y. Chen. Clarabel: An interior-point solver for conic programs with quadratic objectives. arXiv:2405.12762, 2024.

[17] M. Hintermüller, K. Ito, K. Kunisch. The primal–dual active set strategy as a semismooth Newton method. *SIAM J. Optim.* 13 (2002) 865–888.

[18] E. Hairer, C. Lubich, G. Wanner. *Geometric Numerical Integration*, 2nd ed. Springer, 2006.

[19] R. Jordan, D. Kinderlehrer, F. Otto. The variational formulation of the Fokker–Planck equation. *SIAM J. Math. Anal.* 29 (1998) 1–17.

[20] C. Léonard. A survey of the Schrödinger problem and some of its connections with optimal transport. *Discrete Contin. Dyn. Syst.* 34 (2014) 1533–1574.

[21] W. Li, P. Yin, S. Osher. Computations of optimal transport distance with Fisher information regularization. *J. Sci. Comput.* 75 (2018) 1581–1595.

[22] M. S. Lobo, L. Vandenberghe, S. Boyd, H. Lebret. Applications of second-order cone programming. *Linear Algebra Appl.* 284 (1998) 193–228.

[23] W. Li. Transport information geometry: Riemannian calculus on probability simplex. *Inf. Geom.* 5 (2022) 161–207.

[24] J. Maas. Gradient flows of the entropy for finite Markov chains. *J. Funct. Anal.* 261 (2011) 2250–2292.

[25] MOSEK ApS. *MOSEK Modeling Cookbook* (power-cone modeling). Online.

[26] Y. Nesterov, A. Nemirovskii. *Interior-Point Polynomial Algorithms in Convex Programming.* SIAM, 1994.

[27] Q. Mérigot, A. Delalande, F. Chazal. Quantitative stability of optimal transport maps and linearization of the 2-Wasserstein space. *AISTATS* 2020.

[28] T. Pock, A. Chambolle. Diagonal preconditioning for first order primal–dual algorithms in convex optimization. *ICCV* 2011.

[29] U. Ascher, R. Mattheij, R. Russell. *Numerical Solution of Boundary Value Problems for ODEs.* SIAM, 1995.

[30] J. Solomon, R. Rustamov, L. Guibas, A. Butscher. Continuous-flow graph transportation distances. arXiv:1603.06927, 2016.

[31] J. Stoer, R. Bulirsch. *Introduction to Numerical Analysis*, 3rd ed. Springer, 2002.

[32] F. Otto. The geometry of dissipative evolution equations: the porous medium equation. *Comm. PDE* 26 (2001) 101–174.

[33] W. Wang, D. Slepčev, S. Basu, J. A. Ozolek, G. K. Rohde. A linear optimal transportation framework for quantifying and visualizing variations in sets of images. *Int. J. Comput. Vis.* 101 (2013) 254–269.

*Also relevant from the authors' own bibliography: Agueh–Carlier (SIMA 2011) for barycenters; Werenski et al. (ICML 2022) for the analysis QP; Gigli–Maas (SIMA 2013) and Gladbach–Kopfer–Maas (SIMA 2020) for Gromov–Hausdorff consistency; Papadakis–Peyré–Oudet (SIIMS 2014) for proximal splitting of continuous Benamou–Brenier.*

**A note on citation hygiene.** Bibliographic details above were written from memory and cross-checked only partially; before circulating, verify volume/page data, and in particular confirm the exact venue/year of [21] and the current publication status of the arXiv items [2b, 3a, 9a, 16, 30].
