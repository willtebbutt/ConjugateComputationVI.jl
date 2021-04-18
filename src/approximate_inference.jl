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
    return posterior(f(x, σ² .+ 1e-6), y)
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
    m, v = mean_and_var(f_post_approx(x))

    # Compute both of the expectation parameters.
    m1, m2 = expectation_from_canonical(m, v)

    # Compute the gradient w.r.t. both of the expectation parameters. This is equivalent to
    # the natural gradient w.r.t. the natural parameters.
    g1, g2 = Zygote.gradient((m1, m2) -> r(canonical_from_expectation(m1, m2)...), m1, m2)

    # Perform a step of NGA in the first natural pseudo-observation vector.
    η1_new = (1 - ρ) .* η1 .+ ρ .* g1
    η2_new = (1 - ρ) .* η2 .+ ρ .* g2

    return η1_new, η2_new, sqrt(sum(abs2, g1 - η1) + sum(abs2, g2 - η2))
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
    η1, η2, delta_norm = update_approx_posterior(f, x, η1, η2, r, ρ)
    iteration = 1

    # Iterate further until convergence met or max iterations exceeded.
    while delta_norm > tol && iteration < max_iterations
        η1, η2, delta_norm = update_approx_posterior(f, x, η1, η2, r, ρ)
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
