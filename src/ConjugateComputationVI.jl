module ConjugateComputationVI

using AbstractGPs

using AbstractGPs: AbstractGP

"""
    natural_pseudo_obs(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
"""
function natural_pseudo_obs(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    return y ./ σ², (-1) ./ (2 .* σ²)
end

"""
    canonical_pseudo_obs(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})
"""
function canonical_pseudo_obs(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})
    σ² = -1 ./ (2 .* η2)
    y = σ² .* η1
    return y, σ²
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
    y, σ² = canonical_pseudo_obs(η1, η2)
    return posterior(f(x, σ²), y)
end

end
