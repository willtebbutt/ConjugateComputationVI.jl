module ConjugateComputationVI

using AbstractGPs

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

function update_approx_posterior(
    f::AbstractGP,
    x::AbstractVector,
    η1::AbstractVector{<:Real},
    η2::AbstractVector{<:Real},
    ∇r,
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
    g1, g2 = ∇r(m1, m2)

    # Perform a step of gradient ascent in the natural pseudo observations.
    η1_new = (1 - ρ) .* η1 .+ ρ .* g1
    η2_new = (1 - ρ) .* η2 .+ ρ .* g2

    return η1_new, η2_new
end

end
