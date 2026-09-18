# Design notes and reviews

Working documents from the 2026 rewrite of the package (SOCP and Hamiltonian-shooting
solvers, unified `method=` API, admissible means). They record the reasoning behind
decisions and are not maintained as documentation; the docs site and docstrings are
authoritative.

| file | what it is |
|---|---|
| `SPEC.md` | The original implementation spec for the SOCP and shooting rewrite (modules 0–5); "Module N / §" references elsewhere in the history point here. |
| `five_routes_report.md` | Survey of five ways to compute geodesics, distances and barycenters fast, written before the spec. |
| `SOCP_ANALYSIS_SPEC.md` | The analysis (coordinate recovery) fix via the SOCP's endpoint duals, with results (§8). |
| `analysis_via_socp_duals.md` | Why the endpoint-potential Gram matrix is the right object for `analysis`, and what the earlier bug was. |
| `SHOOTING_REVIEW_NOTES.md` | Q&A from the review of the shooting PR: the Hamiltonian formulation, sign conventions, Newton in `log_map`, relation to intrinsic gradient descent. |
| `pr23_review.md` | The review itself (requested changes, description gaps, coverage notes). |
| `SINKHORN_METHOD_DESIGN.md` | How the Sinkhorn code was folded into the unified API. |
| `ADMISSIBLE_MEANS_DESIGN.md` | Conic forms for the harmonic and (quadrature) logarithmic means, the ordering of the means, theory flags, and the PR plan. |
