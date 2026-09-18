# References

The package computes with the discrete transport metric on a reversible Markov chain,
introduced independently by Maas and by Chow, Huang, Li and Zhou, and with the
geodesics, barycenters and barycentric coordinates that this geometry supports. This
page lists the literature behind each part of the package.

## Citing this package

The package accompanies

- D. Gentile, J. M. Murphy. *Static and dynamic approaches to computing barycenters of
  probability measures on graphs.* Preprint, 2026.

which introduces the discrete transport barycenter on graphs, its computation by
intrinsic gradient descent and by a single conic program, and the recovery of
barycentric coordinates. An earlier, entropically regularized treatment is

- D. Gentile, J. M. Murphy. Regularized 2-Wasserstein barycenters on discrete metric
  spaces. *2025 International Conference on Sampling Theory and Applications (SampTA)*,
  IEEE, 2025, pp. 1–5.

## The discrete transport metric

The metric, its Benamou–Brenier form and its dependence on an admissible mean
(`MarkovGraph`, `AdmissibleMean`):

- J. Maas. Gradient flows of the entropy for finite Markov chains. *Journal of Functional
  Analysis* 261 (8), 2011, 2250–2292. The metric with the logarithmic mean, for which the
  heat flow is the gradient flow of the entropy (tested in `test/runtests.jl`).
- S.-N. Chow, W. Huang, Y. Li, H. Zhou. Fokker–Planck equations for a free energy
  functional or Markov process on a graph. *Archive for Rational Mechanics and Analysis*
  203 (3), 2012, 969–1008.
- A. Mielke. Geodesic convexity of the relative entropy in reversible Markov chains.
  *Calculus of Variations and Partial Differential Equations* 48, 2013, 1–31. General
  admissible means.
- M. Erbar, J. Maas. Ricci curvature of finite Markov chains via convexity of the entropy.
  *Archive for Rational Mechanics and Analysis* 206 (3), 2012, 997–1038.
- M. Erbar, J. Maas, M. Wirth. On the geometry of geodesics in discrete optimal transport.
  *Calculus of Variations and Partial Differential Equations* 58 (1), 2019, 19. Geodesics
  and the Hamiltonian structure behind `exp_map` / `log_map`.
- N. Gigli, J. Maas. Gromov–Hausdorff convergence of discrete transportation metrics.
  *SIAM Journal on Mathematical Analysis* 45 (2), 2013, 879–899.
- P. Gladbach, E. Kopfer, J. Maas. Scaling limits of discrete optimal transport. *SIAM
  Journal on Mathematical Analysis* 52 (3), 2020, 2759–2802.
- J. Solomon, R. Rustamov, L. Guibas, A. Butscher. Continuous-flow graph transportation
  distances. arXiv:1603.06927, 2016.

## Computing geodesics

- J.-D. Benamou, Y. Brenier. A computational fluid mechanics solution to the
  Monge–Kantorovich mass transfer problem. *Numerische Mathematik* 84 (3), 2000, 375–393.
  The dynamic formulation that the SOCP discretizes in time (`geodesic`, `method=:socp`).
- M. Erbar, M. Rumpf, B. Schmitzer, S. Simon. Computation of optimal transport on
  discrete metric measure spaces. *Numerische Mathematik* 144 (1), 2020, 157–200. The
  Galerkin time discretization and the primal-dual solver kept as
  `method=:chambolle_pock`; also the source of the geometric-mean default.
- A. Chambolle, T. Pock. A first-order primal-dual algorithm for convex problems with
  applications to imaging. *Journal of Mathematical Imaging and Vision* 40 (1), 2011,
  120–145.
- P. Goulart, Y. Chen. Clarabel: An interior-point solver for conic programs with
  quadratic objectives. arXiv:2405.12762, 2024. The conic solver behind `method=:socp`.
- M. Lubin, O. Dowson, J. Dias Garcia, J. Huchette, B. Legat, J. P. Vielma. JuMP 1.0:
  recent improvements to a modeling language for mathematical optimization.
  *Mathematical Programming Computation* 15, 2023, 581–589.

## Barycenters and barycentric coordinates

- M. Agueh, G. Carlier. Barycenters in the Wasserstein space. *SIAM Journal on
  Mathematical Analysis* 43 (2), 2011, 904–924.
- Y.-H. Kim, B. Pass. Wasserstein barycenters over Riemannian manifolds. *Advances in
  Mathematics* 307, 2017, 640–683.
- P. C. Álvarez-Esteban, E. del Barrio, J. A. Cuesta-Albertos, C. Matrán. A fixed-point
  approach to barycenters in Wasserstein space. *Journal of Mathematical Analysis and
  Applications* 441 (2), 2016, 744–762. The fixed-point view of the Riemannian descent in
  `barycenter(method=:shooting)`.
- J. M. Altschuler, E. Boix-Adsera. Wasserstein barycenters are NP-hard to compute. *SIAM
  Journal on Mathematics of Data Science* 4 (1), 2022, 179–203.
- M. Werenski, R. Jiang, A. Tasissa, S. Aeron, J. M. Murphy. Measure estimation in the
  barycentric coding model. *Proceedings of the 39th International Conference on Machine
  Learning*, PMLR, 2022, pp. 23781–23803. The barycentric coding model and the Gram-matrix
  quadratic program used by `analysis`.
- M. Werenski, B. Mallery, S. Aeron, J. M. Murphy. Linearized Wasserstein barycenters:
  synthesis, analysis, representational capacity, and applications. *Proceedings of the
  28th International Conference on Artificial Intelligence and Statistics*, PMLR, 2025.
- B. Mallery, J. M. Murphy, S. Aeron. Synthesis and analysis of data as probability
  measures with entropy-regularized optimal transport. *Proceedings of the 28th
  International Conference on Artificial Intelligence and Statistics*, PMLR, 2025,
  pp. 2584–2592.

## Entropic transport

- R. Sinkhorn, P. Knopp. Concerning nonnegative matrices and doubly stochastic matrices.
  *Pacific Journal of Mathematics* 21 (2), 1967, 343–348.
- M. Cuturi. Sinkhorn distances: lightspeed computation of optimal transport. *Advances in
  Neural Information Processing Systems* 26, 2013.
- J.-D. Benamou, G. Carlier, M. Cuturi, L. Nenna, G. Peyré. Iterative Bregman projections
  for regularized transportation problems. *SIAM Journal on Scientific Computing* 37 (2),
  2015, A1111–A1138. Entropic barycenters (`method=:sinkhorn`).
- N. Bonneel, G. Peyré, M. Cuturi. Wasserstein barycentric coordinates: histogram
  regression using optimal transport. *ACM Transactions on Graphics* 35 (4), 2016, 1–10.
  The differentiated Sinkhorn iteration in `analysis(method=:sinkhorn)`.
- H. Janati, M. Cuturi, A. Gramfort. Debiased Sinkhorn barycenters. *Proceedings of the
  37th International Conference on Machine Learning*, PMLR, 2020, pp. 4692–4701.
- G. Peyré, M. Cuturi. Computational optimal transport. *Foundations and Trends in Machine
  Learning* 11 (5–6), 2019, 355–607.

## Background

- R. Jordan, D. Kinderlehrer, F. Otto. The variational formulation of the Fokker–Planck
  equation. *SIAM Journal on Mathematical Analysis* 29 (1), 1998, 1–17.
- L. Ambrosio, N. Gigli, G. Savaré. *Gradient Flows in Metric Spaces and in the Space of
  Probability Measures.* Birkhäuser, 2005.
- C. Villani. *Optimal Transport: Old and New.* Grundlehren der mathematischen
  Wissenschaften 338, Springer, 2009.
- F. Santambrogio. *Optimal Transport for Applied Mathematicians.* Progress in Nonlinear
  Differential Equations and Their Applications 87, Birkhäuser, 2015.
