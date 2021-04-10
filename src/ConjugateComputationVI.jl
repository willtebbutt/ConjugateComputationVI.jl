module ConjugateComputationVI

using AbstractGPs
using Distributions
using FastGaussQuadrature
using Zygote

using AbstractGPs: AbstractGP

"""
    natural_from_canonical(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
"""
function natural_from_canonical(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    return y ./ σ², (-1) ./ (2 .* σ²)
end

"""
    canonical_from_natural(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})
"""
function canonical_from_natural(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})
    σ² = -1 ./ (2 .* η2)
    y = σ² .* η1
    return y, σ²
end

"""
    expectation_from_canonical(m::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
"""
function expectation_from_canonical(m::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    return m, m.^2 .+ σ²
end

"""
    canonical_from_expectation(m1::AbstractVector{<:Real}, m2::AbstractVector{<:Real})
"""
function canonical_from_expectation(m1::AbstractVector{<:Real}, m2::AbstractVector{<:Real})
    return m1, m2 .- m1.^2
end

"""
    gaussian_reconstruction_term(
        y::AbstractVector{<:Real},
        σ²::AbstractVector{<:Real},
        m̃::AbstractVector{<:Real},
        σ̃²::AbstractVector{<:Real},
    )
"""
function gaussian_reconstruction_term(
    y::AbstractVector{<:Real},
    σ²::AbstractVector{<:Real},
    m̃::AbstractVector{<:Real},
    σ̃²::AbstractVector{<:Real},
)
    return sum(
        map((y, σ², m̃, σ̃²) -> logpdf(Normal(m̃, sqrt(σ²)), y) - σ̃² / (2σ²), y, σ², m̃, σ̃²),
    )
end

"""
    approx_posterior(
        f::AbstractGP,
        x::AbstractVector,
        η1::AbstractVector{<:Real},
        η2::AbstractVector{<:Real},
    )

Compute the approximate posterior. This is just a posterior GP some kind, which is another
AbstractGP.
"""
function approx_posterior(
    f::AbstractGP,
    x::AbstractVector,
    η1::AbstractVector{<:Real},
    η2::AbstractVector{<:Real},
)
    y, σ² = canonical_from_natural(η1, η2)
    return posterior(f(x, σ²), y)
end

"""
    update_approx_posterior(
        f::AbstractGP,
        x::AbstractVector,
        η1::AbstractVector{<:Real},
        η2::AbstractVector{<:Real},
        r,
        ρ::Real,
    )
"""
function update_approx_posterior(
    f::AbstractGP,
    x::AbstractVector,
    η1::AbstractVector{<:Real},
    η2::AbstractVector{<:Real},
    r,
    ρ::Real,
)
    # Check that the step size makes sense.
    (ρ > 1 || ρ <= 0) && throw(error("Bad step size"))

    # Compute the approximate posterior.
    f_post_approx = approx_posterior(f, x, η1, η2)
    ms = marginals(f_post_approx(x))

    # Compute both of the expectation parameters.
    m1, m2 = expectation_from_canonical(mean.(ms), var.(ms))

    # Compute the gradient w.r.t. both of the expectation parameters. This is equivalent to
    # the natural gradient w.r.t. the natural parameters.
    g1, g2 = Zygote.gradient((m1, m2) -> r(canonical_from_expectation(m1, m2)...), m1, m2)

    # Perform a step of gradient ascent in the natural pseudo observations.
    η1_new = (1 - ρ) .* η1 .+ ρ .* g1
    η2_new = (1 - ρ) .* η2 .+ ρ .* g2

    return η1_new, η2_new, g1, g2
end

"""
    optimise_approx_posterior(
        f::AbstractGP,
        x::AbstractVector,
        η1::AbstractVector{<:Real},
        η2::AbstractVector{<:Real},
        ∇r,
        ρ::Real;
        max_iterations=1_000,
        gtol=1e-8,
    )
"""
function optimise_approx_posterior(
    f::AbstractGP,
    x::AbstractVector,
    η1::AbstractVector{<:Real},
    η2::AbstractVector{<:Real},
    r,
    ρ::Real;
    max_iterations=1_000,
    tol=1e-8,
)
    # Perform initial iteration.
    η1, η2, g1, g2 = update_approx_posterior(f, x, η1, η2, r, ρ)
    delta_norm = sqrt(sum(abs2, η1 - g1) + sum(abs2, η2 - g2))
    iteration = 0

    # Iterate further until convergence met or max iterations exceeded.
    while delta_norm > tol && iteration < max_iterations
        η1, η2, g1, g2 = update_approx_posterior(f, x, η1, η2, r, ρ)
        delta_norm = sqrt(sum(abs2, η1 - g1) + sum(abs2, η2 - g2))
        iteration += 1
    end
    return η1, η2, iteration, delta_norm
end

"""
    elbo(
        f::AbstractGP,
        x::AbstractVector,
        η1::AbstractVector{<:Real},
        η2::AbstractVector{<:Real},
        r,
    )
"""
function AbstractGPs.elbo(
    f::AbstractGP,
    x::AbstractVector,
    η1::AbstractVector{<:Real},
    η2::AbstractVector{<:Real},
    r,
)
    ỹ, σ̃² = canonical_from_natural(η1, η2)

    # Compute reconstruction term under Gaussian pseudo-likelihood.
    approx_post_marginals = marginals(approx_posterior(f, x, η1, η2)(x))
    mq = mean.(approx_post_marginals)
    σ²q = var.(approx_post_marginals)
    return logpdf(f(x, σ̃²), ỹ) + r(mq, σ²q) - gaussian_reconstruction_term(ỹ, σ̃², mq, σ²q)
end

# I generally expect that num_points is quite small, but length(fs) could be quite large.
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

end
