# Solving the Analysis Problem via SOCP Duals

# The barycentric-coordinate recovery bug: what was wrong, why the fix is correct
   
## 1. Setting and notation

Let $(X, Q, \pi)$ be a finite graph with an irreducible, reversible Markov kernel $Q$ and stationary distribution $\pi$. Following Maas [1] and Erbar–Maas [2], the discrete transport distance between densities $\rho_0, \rho_1$ (w.r.t. $\pi$) is

$$
\mathcal{W}(\rho_0,\rho_1)^2 = \inf \left\{ \int_0^1 \frac12 \sum_{x,y} \frac{m_t(x,y)^2}{\theta(\rho_t(x),\rho_t(y))}\, Q(x,y)\pi(x)\, dt \;:\; \dot\rho_t + \operatorname{div} m_t = 0,\ \rho(0)=\rho_0,\ \rho(1)=\rho_1 \right\}
$$

with $\theta$ the geometric mean throughout this work. Equivalently, in terms of potentials $\varphi$ with $m = \theta(\rho)\circ\nabla\varphi$, the metric tensor at $\rho$ is

$$
\langle \nabla\varphi, \nabla\psi\rangle_\rho = \tfrac12 \sum_{x,y} \theta(\rho(x),\rho(y))\, \nabla\varphi(x,y)\,\nabla\psi(x,y)\, Q(x,y)\pi(x) = \sum_{e} \kappa_e\, \theta(\rho)_e\, (\nabla\varphi)_e (\nabla\psi)_e .
$$

The barycenter of references $\{\nu_i\}_{i=1}^p$ with weights $\lambda \in \Delta^{p-1}$ is the minimizer of $J(\nu) = \sum_i \lambda_i \mathcal{W}(\nu_i, \nu)^2$ (Agueh–Carlier [3] in the continuous setting).

**Analysis problem.** Given a `target` $\nu$ and references $\nu_i$, recover $\lambda$. The method (from the dissertation, and spec.txt Module 4) forms tangent vectors $v_i = \log_\nu(\nu_i)$, the Gram matrix $A_{ij} = \langle v_i, v_j\rangle_\nu$, and solves

$$
\hat\lambda = \arg\min_{\lambda\in\Delta^{p-1}} \lambda^\top A \lambda = \arg\min_{\lambda\in\Delta^{p-1}} \Big\| \sum_i \lambda_i v_i \Big\|_\nu^2 .
$$

This is justified by the first-order optimality condition of the barycenter problem: at a minimizer $\nu$, $\sum_i \lambda_i \log_\nu(\nu_i) = 0$ (the Riemannian gradient of $\frac12 \mathcal{W}(\nu_i,\cdot)^2$ at $\nu$ is $-\log_\nu(\nu_i)$, a standard fact on Riemannian manifolds; see e.g. Karcher [4] for the Riemannian center-of-mass setting and Agueh–Carlier [3] for the Wasserstein analogue). So the true $\lambda$ is a zero of the quadratic form, and the QP recovers it.

The crucial point, made precise below: **this argument only works if the tangent vectors used in $A$ are the ones in whose stationarity convention $\nu$ actually is stationary.**

## 2. The bug

### 2.1 Symptom

`analyze_socp` recovered the weights of an SOCP-synthesized barycenter on the 160-node Massachusetts House graph at $N=2$ with 27.6% relative error, while the same QP applied to the descent-synthesized barycenter recovered 0.38%. A 2×2 cross test (both target points × both geodesic solvers) showed the *solver* was irrelevant and the *target point* was everything. The residual diagnostic $\lambda^\top A\lambda / \hat\lambda^\top A\hat\lambda$ was 1.95 at $\nu_{\text{socp}}$ (the true $\lambda$ was far from a zero of the form) versus 1.02 at $\nu_{\text{descent}}$.

### 2.2 Cause

The original `analyze_socp` used, as the tangent vector for reference $i$, the **initial momentum** $m_i[:,1]$ of the discrete geodesic SOCP from $\nu$ to $\nu_i$, with the reasoning "$m = \theta\circ\nabla\varphi$, so momenta and potential gradients are equivalent." That identity is true in the continuum but **not at the level of the time-discretized program**.

The discrete geodesic SOCP (spec.txt Module 1, after Erbar–Maas's Benamou–Brenier form, cf. [2, 5]) has variables $\rho_t$ at $N+1$ time nodes and $m_t$ on the $N$ intervals, with the continuity equation $(\rho_{t+1}-\rho_t)/h + \operatorname{div} m_t = 0$ and the action evaluated at the interval **midpoint** density $\bar\rho_t = (\rho_t + \rho_{t+1})/2$. Writing the KKT conditions with multiplier $\psi_t$ on the continuity equation at interval $t$, stationarity in $m_t$ gives (verified numerically to solver tolerance)

$$
m_t = -\frac{1}{2h}\, \theta(\bar\rho_t)\circ \nabla\!\left(\psi_t/\pi\right).
$$

So $m_1$ is $\theta$ *at the midpoint of the first interval* times the gradient of the *half-step* potential. It is an $O(h)$ approximation to $\theta(\nu)\circ\nabla\varphi(0)$, and at $N=2$ the error is not small. The earlier abandoned attempt to read $\varphi$ off the continuity duals found "no clean constant" precisely because it paired $\psi_1$ with $\theta(\nu)$ instead of $\theta(\bar\rho_1)$.

The deeper problem is not the $O(h)$ approximation per se. It is that $\nu_{\text{socp}}$, being the KKT point of the *joint* SOCP, satisfies a stationarity condition in **its own** discrete potentials, and $\sum_i \lambda_i m_i[:,1] = 0$ is not that condition. So the true $\lambda$ was not a zero of the Gram form, and the QP found a different $\hat\lambda$ that fit the wrong tangent vectors better.

## 3. The fix and its mathematical basis

### 3.1 The discrete potential is the endpoint-constraint dual

Let $V_h(\rho_0, \rho_1)$ denote the optimal value of the discrete geodesic SOCP, i.e. $\mathcal{W}_h(\rho_0,\rho_1)^2$. The endpoint constraint $\rho[:,1] = \rho_0$ enters as $n$ equality constraints; by standard convex sensitivity analysis (Lagrangian duality / envelope theorem, e.g. Boyd–Vandenberghe [6, §5.6], Rockafellar [7]), the Lagrange multiplier $y_0 \in \mathbb{R}^n$ of that constraint is the gradient of the value function with respect to the constraint's right-hand side:

$$
\delta V_h = \langle y_0, \delta\rho_0\rangle_{\mathbb{R}^n} = \langle y_0/\pi,\ \delta\rho_0\rangle_\pi .
$$

Define $\varphi_0 := y_0/\pi$. Then $\varphi_0$ is exactly $\nabla_{\rho_0} \mathcal{W}_h^2$ under the $\pi$-weighted pairing, which is the discrete analogue of the Hamilton–Jacobi / Kantorovich-potential identity $\partial_{\rho_0}\frac12\mathcal{W}^2 = -\varphi(0)$ (Otto–Villani [8]; Villani [9, Thm 8.13]; in the graph setting, Erbar–Maas [2, §3] and the Hamiltonian formulation of Chow–Li–Zhou [10]). The identity is well defined only up to an additive constant in $\varphi_0$ (the value function lives on the hyperplane $\langle\rho_0,\pi\rangle=1$), which is harmless since only $\nabla\varphi_0$ enters $A$.

Empirical calibration (required because JuMP/Clarabel dual signs are solver-convention dependent): $\varphi_0$ from the dual matched a central finite difference of $V_h$ to $\sim 10^{-5}$ and the two-node closed form's derivative $\frac{d}{ds}\mathcal{W}(\rho(s),\rho(t))^2 = -\sqrt2\,\mathcal{W}\,(1-s^2)^{-1/4}$ with $O(h)$ error. No sign flip was needed.

### 3.2 Exact discrete stationarity of the SOCP barycenter

The joint barycenter SOCP is

$$
\min_{\nu,\{\rho^i,m^i\}} \sum_i \lambda_i\, \text{Action}_h(\rho^i, m^i) \quad\text{s.t.}\quad \rho^i[:,N+1] = \nu\ \ \forall i,\quad \langle \nu,\pi\rangle = 1,\quad \nu \ge 0,
$$

plus each block's own continuity and cone constraints. Let $y^i_1$ be the multiplier of block $i$'s shared-endpoint constraint. Because block $i$'s action is weighted by $\lambda_i$, $y^i_1 = \lambda_i \,\pi\circ\varphi^i_1$ where $\varphi^i_1$ is the endpoint potential of the *unweighted* geodesic from $\nu_i$ to $\nu$. Stationarity of the Lagrangian in $\nu$ reads

$$
\sum_i \lambda_i\, \pi\circ\varphi^i_1 \;=\; \mu\,\pi + s, \qquad s \ge 0,\ s\circ\nu = 0,
$$

with $\mu$ the multiplier of the normalization. On the support of $\nu$ (in practice all of $X$ for the instances considered) this is

$$
\boxed{\ \sum_i \lambda_i\,\varphi^i_1 = \text{const} \quad\Longrightarrow\quad \sum_i \lambda_i\,\nabla\varphi^i_1 = 0\ }
$$

**exactly**, to solver tolerance, at any $N$ — including $N=2$. This was verified directly: the gradient of $\sum_i\lambda_i\varphi^i_1$ is below $10^{-6}$ relative on test instances.

By time-reversal symmetry of the discrete program (the continuity equation with $m\mapsto -m$ and the midpoint density are both invariant under $t \mapsto 1-t$), $\mathcal{W}_h(a,b)=\mathcal{W}_h(b,a)$ and the right-endpoint dual of the geodesic $\nu_i\to\nu$ equals the left-endpoint dual of the geodesic $\nu\to\nu_i$. So an *independent* solve from `target` to each reference, which is what an analysis routine must do for an arbitrary target, produces exactly the potentials in which $\nu_{\text{socp}}$ is stationary.

### 3.3 The corrected Gram matrix

`analyze_socp` now forms, per spec.txt Module 4,

$$
A_{ij} = \sum_e \kappa_e\, \theta(\nu)_e\, (\nabla\varphi^i_0)_e\, (\nabla\varphi^j_0)_e ,
$$

with $\varphi^i_0$ the endpoint dual of the geodesic SOCP from $\nu$ to $\nu_i$, and solves the same simplex QP. By §3.2, when $\nu$ came from `barycenter_socp` at the same $N$, the true $\lambda$ satisfies $\sum_i\lambda_i\nabla\varphi^i_0 = 0$, hence $\lambda^\top A\lambda = 0$ up to solver noise. Since $A \succeq 0$, $\lambda$ is a global minimizer of the QP; if the $\nabla\varphi^i_0$ span a $(p-1)$-dimensional space (generic), $\lambda$ is the unique minimizer on the simplex. Note that the weighting $\kappa_e\theta(\nu)_e$ is the natural Riemannian one but is not essential to correctness: any positive weighting has the same zero set. It matters only for conditioning.

### 3.4 Numerical confirmation

| target | tangent convention | $\hat\lambda$ | rel. err |
|---|---|---|---|
| $\nu_{\text{socp}}$ (N=2) | endpoint potential (new) | [0.4000, 0.3000, 0.2000, 0.1000] | $1.3\times10^{-5}$ |
| $\nu_{\text{socp}}$ (N=2) | momentum (old) | [0.294, 0.408, 0.199, 0.099] | 27.6% |
| $\nu_{\text{descent}}$ | momentum (old) | [0.400, 0.301, 0.200, 0.098] | 0.38% |
| $\nu_{\text{descent}}$ | endpoint potential (new) | [0.474, 0.215, 0.216, 0.095] | 20.8% |

At $\nu_{\text{socp}}$ with the native Gram matrix, $\lambda^\top A\lambda \approx 4\times10^{-8}$ against $\max\operatorname{diag}A \approx 140$. Across 56 grid/proximity round-trip cells the worst error dropped from 18.7% to $3.3\times10^{-4}$ and became flat in $N$, graph size, and reference separation.

The bottom row is the same phenomenon in mirror image and is the cleanest statement of the lesson: **a synthesized barycenter is stationary only in the discrete convention that synthesized it.** The Chambolle–Pock descent barycenter satisfies $\sum_i\lambda_i m_i^{\text{CP}}(0)\approx 0$ in CP's momentum convention (its literal stopping criterion) and is *not* a zero of the SOCP-potential form. Likewise, analyzing an $N=2$ SOCP barycenter with $N=10$ geodesics degrades to 3.5%: the discretization parameter is part of the convention. All conventions agree as $h\to0$, but at practical $N$ the analysis must use the synthesis's own first-order condition. This will apply equally to the Hamiltonian-shooting backend, which is a third discretization (RK4 flow of the Chow–Li–Zhou / Erbar–Maas Hamiltonian [2, 10]).

## 4. Things still to nail down for a paper

- A written derivation of the full KKT system of the discrete geodesic SOCP (continuity multipliers $\psi_t$, mean-cone and action-epigraph RSOC multipliers, endpoint multipliers), showing (a) $m_t = -\frac{1}{2h}\theta(\bar\rho_t)\nabla(\psi_t/\pi)$ and (b) the relation between the endpoint multiplier and $\psi_1/h$ with the explicit RSOC correction terms that vanish as $h\to0$. Both are currently verified numerically, not proved on paper.
- Uniqueness of the QP minimizer (generic linear independence of $\{\nabla\varphi^i_0\}$ modulo the one relation), and a statement about the boundary case where $\nu$ has zero entries and the slack $s$ appears.
- The dual-sign convention should be stated as a calibrated fact (Clarabel via JuMP/MathOptInterface returns the multiplier so that $\delta V = \langle y,\delta b\rangle$ for a minimization) rather than assumed.
- Whether to treat the $O(h)$ agreement between conventions as a theorem (consistency of the discretizations) or leave it empirical.

## References

1. J. Maas, *Gradient flows of the entropy for finite Markov chains*, J. Funct. Anal. 261 (2011) 2250–2292.
2. M. Erbar, J. Maas, *Ricci curvature of finite Markov chains via convexity of the entropy*, Arch. Ration. Mech. Anal. 206 (2012) 997–1038. (The paper the codebase's "Erbar et al." framework, Lemma 2.4 antisymmetry, and Benamou–Brenier form follow; the Erbar–Maas–Wirth computational paper below is the one with Fig. 5/9 phenomenology referenced in spec.txt.)
3. M. Agueh, G. Carlier, *Barycenters in the Wasserstein space*, SIAM J. Math. Anal. 43 (2011) 904–924.
4. H. Karcher, *Riemannian center of mass and mollifier smoothing*, Comm. Pure Appl. Math. 30 (1977) 509–541.
5. J.-D. Benamou, Y. Brenier, *A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem*, Numer. Math. 84 (2000) 375–393; and M. Erbar, J. Maas, M. Wirth, *On the geometry of geodesics in discrete optimal transport*, Calc. Var. PDE 58 (2019), for the discrete geodesic computations the SOCP mirrors.
6. S. Boyd, L. Vandenberghe, *Convex Optimization*, Cambridge Univ. Press, 2004, §5.6 (perturbation and sensitivity analysis).
7. R. T. Rockafellar, *Convex Analysis*, Princeton Univ. Press, 1970 (Lagrange multipliers as subgradients of the value function).
8. F. Otto, C. Villani, *Generalization of an inequality by Talagrand and links with the logarithmic Sobolev inequality*, J. Funct. Anal. 173 (2000) 361–400.
9. C. Villani, *Topics in Optimal Transportation*, AMS GSM 58, 2003, Ch. 8 (Hamilton–Jacobi / first variation of $\frac12 W_2^2$).
10. S.-N. Chow, W. Li, H. Zhou, *A discrete Schrödinger equation via optimal transport on graphs* / *Entropy dissipation of Fokker–Planck equations on graphs*, and W. Li's Hamiltonian-flow formulation of discrete OT (e.g. Chow, Li, Zhou, *Wasserstein Hamiltonian flows*, J. Diff. Eq. 268 (2020)), for the $(\rho,\varphi)$ Hamiltonian system used in Module 3.
11. A. Chambolle, T. Pock, *A first-order primal-dual algorithm for convex problems with applications to imaging*, J. Math. Imaging Vis. 40 (2011) 120–145 (the incumbent geodesic solver whose stationarity convention the descent barycenter satisfies).

Citation details (volumes, pages, and which Erbar–Maas paper is the right one for a given claim) should be checked against the actual bibliography before submission; I've given them from memory and can't verify them here.

