"""
    AdmissibleMean

Abstract supertype of the means `θ(s, t)` (mobilities) that define a discrete transport
metric: `‖∇φ‖²_ρ = Σ_e κ_e θ(ρ_x, ρ_y) (∇φ)_e²`. An admissible mean is continuous,
symmetric, positively 1-homogeneous, concave, positive on `(0,∞)²` and normalised so
`θ(s, s) = s` (Maas 2011). Concavity is what makes the action `m²/θ` jointly convex,
hence the geodesic/barycenter programs convex.

Every concrete mean `θ` is callable, `θ(s, t)`, and has `partial_s(θ, s, t)` (the
partial derivative in the first argument; `partial_t` follows by symmetry). Both are
generic in the element type so `ForwardDiff` can differentiate through them.

For all `s, t > 0`: `HarmonicMean() ≤ GeometricMean() ≤ LogarithmicMean() ≤ ArithmeticMean()`.
The arithmetic mean is the only one with `θ(0, t) ≠ 0`, so it is the only one under which
mass can flow out of an empty node; see the package documentation on boundary behaviour.
"""
abstract type AdmissibleMean end

"""`θ(s,t) = √(st)`. The package's default, following Erbar et al. (2020)."""
struct GeometricMean <: AdmissibleMean end
"""`θ(s,t) = (s+t)/2`. Does not vanish at an empty node (`θ(0,t) = t/2`)."""
struct ArithmeticMean <: AdmissibleMean end
"""`θ(s,t) = 2st/(s+t)`. The smallest admissible mean of the four; bounded `∂₁θ` near zero."""
struct HarmonicMean <: AdmissibleMean end
"""
`θ(s,t) = (s−t)/(ln s − ln t)` (Λ). The mean for which the heat flow is the gradient flow
of the relative entropy (Maas 2011). Evaluated by a series near `s = t` so that the value
and its derivative are smooth there (needed for `ForwardDiff` through the shooting maps).
Not finitely conic-representable; the SOCP uses `QuadLogMean` instead.
"""
struct LogarithmicMean <: AdmissibleMean end

"""
    QuadLogMean(K::Int=8)

Gauss–Legendre approximation of the logarithmic mean,
`Λ_K(s,t) = Σ_k w_k s^{α_k} t^{1−α_k} ≈ ∫₀¹ s^α t^{1−α} dα = Λ(s,t)`, with `K` nodes.
`Λ_K` is itself an admissible mean (symmetric, homogeneous, concave, positive), so a
transport metric built on it is exact, not approximate; and each term is a power-cone
constraint, which is how the SOCP represents the logarithmic mean. Worst relative error
against `Λ` over density ratios up to 1000: `8.5e-4` (K=4), `6e-7` (K=6), `1.2e-10` (K=8),
`2e-15` (K=12).
"""
struct QuadLogMean <: AdmissibleMean
    α::Vector{Float64}
    w::Vector{Float64}
    function QuadLogMean(K::Int=8)
        K ≥ 1 || throw(ArgumentError("QuadLogMean needs K ≥ 1 nodes"))
        # Golub–Welsch: Gauss–Legendre nodes/weights on [-1,1], mapped to [0,1]
        if K == 1
            return new([0.5], [1.0])
        end
        J = SymTridiagonal(zeros(K), [k / sqrt(4k^2 - 1) for k in 1:K-1])
        E = eigen(J)
        α = (E.values .+ 1) ./ 2
        w = E.vectors[1, :] .^ 2            # Σw = 1 on [0,1]
        return new(α, w ./ sum(w))
    end
end

(::GeometricMean)(s, t)  = sqrt(s * t)
(::ArithmeticMean)(s, t) = (s + t) / 2
(::HarmonicMean)(s, t)   = 2 * s * t / (s + t)
function (::LogarithmicMean)(s, t)
    δ = s / t - 1
    if abs(δ) < 1e-3
        # (x−1)/ln x = 1 + δ/2 − δ²/12 + δ³/24 − 19δ⁴/720 + 3δ⁵/160 + O(δ⁶)
        return t * (1 + δ * (1/2 + δ * (-1/12 + δ * (1/24 + δ * (-19/720 + δ * 3/160)))))
    end
    return t * δ / log1p(δ)          # log1p keeps the closed form accurate down to the switch
end
(θ::QuadLogMean)(s, t) = sum(θ.w[k] * s^θ.α[k] * t^(1 - θ.α[k]) for k in eachindex(θ.α))

"""
    partial_s(θ::AdmissibleMean, s, t)

`∂θ/∂s(s, t)`. By symmetry `partial_t(θ, s, t) = partial_s(θ, t, s)`.
"""
partial_s(::GeometricMean, s, t)  = sqrt(t / s) / 2
partial_s(::ArithmeticMean, s, t) = one(s) / 2
partial_s(::HarmonicMean, s, t)   = 2 * t^2 / (s + t)^2
function partial_s(::LogarithmicMean, s, t)
    δ = s / t - 1
    if abs(δ) < 1e-3
        # d/dx (x−1)/ln x = 1/2 − δ/6 + δ²/8 − 19δ³/180 + 3δ⁴/32 + O(δ⁵)
        return 1/2 + δ * (-1/6 + δ * (1/8 + δ * (-19/180 + δ * 3/32)))
    end
    L = log1p(δ)
    return (L - δ / (1 + δ)) / L^2   # = (ln x − 1 + 1/x)/(ln x)², x = s/t
end
partial_s(θ::QuadLogMean, s, t) = sum(θ.w[k] * θ.α[k] * s^(θ.α[k] - 1) * t^(1 - θ.α[k]) for k in eachindex(θ.α))

partial_t(θ::AdmissibleMean, s, t) = partial_s(θ, t, s)

Base.show(io::IO, θ::QuadLogMean) = print(io, "QuadLogMean(", length(θ.α), ")")
