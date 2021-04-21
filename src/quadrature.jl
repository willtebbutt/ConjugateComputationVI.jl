"""
    batch_quadrature(
        fs::AbstractVector,
        ms::AbstractVector{<:Real},
        σs::AbstractVector{<:Real},
        num_points::Integer,
    )

Approximate the integrals
```julia
∫ fs[n](x) pdf(Normal(ms[n], σs[n]), x) dx
```
for all `n` in `eachindex(fs)` using Gauss-Hermite quadrature with `num_points`.
"""
function batch_quadrature(
    fs::AbstractVector,
    ms::AbstractVector{<:Real},
    σs::AbstractVector{<:Real},
    num_points::Integer,
)
    # Check that as many bounds are provided as we have functions to integrate.
    length(fs) == length(ms) || throw(error("length(fs) != length(ms)"))
    length(fs) == length(σs) || throw(error("length(fs) != length(σs)"))

    # Construct the quadrature points.
    xs, ws = gausshermite(num_points)

    # Compute the integral.
    return map((f, m, σ) -> _gauss_hermite_quadrature(f, m, σ, xs, ws), fs, ms, σs)
end

Zygote.@nograd gausshermite

# Internal method. Assumes that the gradient w.r.t. xs and ws is never needed, so avoids
# computing it and returns nothing. This is potentially not what you want in general.
function _gauss_hermite_quadrature(f, m::Real, σ::Real, xs, ws)
    t(x, m, σ) = m + sqrt(2) * σ * x
    I = ws[1] * f(t(xs[1], m, σ))
    for j in 2:length(xs)
        I += ws[j] * f(t(xs[j], m, σ))
    end
    return I / sqrt(π)
end

function Zygote._pullback(
    ctx::Zygote.AContext, ::typeof(_gauss_hermite_quadrature), f, m::Real, σ::Real, xs, ws,
)
    function _gauss_hermite_quadrature_pullback(Δ::Real)
        g(f, x, w, m, σ) = w * f(m + sqrt(2) * σ * x)

        _, pb = Zygote._pullback(ctx, g, f, xs[1], ws[1], m, σ)
        _, Δf, _, _, Δm, Δσ = pb(Δ / sqrt(π))
        for j in 2:length(xs)
            _, pb = Zygote._pullback(ctx, g, f, xs[j], ws[j], m, σ)
            _, Δf_, _, _, Δm_, Δσ_ = pb(Δ / sqrt(π))
            Δf = Zygote.accum(Δf, Δf_)
            Δm = Zygote.accum(Δm, Δm_)
            Δσ = Zygote.accum(Δσ, Δσ_)
        end
        return nothing, Δf, Δm, Δσ, nothing, nothing
    end
    return _gauss_hermite_quadrature(f, m, σ, xs, ws), _gauss_hermite_quadrature_pullback
end
