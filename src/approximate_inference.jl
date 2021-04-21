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
    mq, σ²q = mean_and_var(approx_posterior(f, x, η1, η2)(x))

    # Compute both of the expectation parameters.
    m1, m2 = expectation_from_canonical(mq, σ²q)

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



abstract type AbstractIntegrater end

"""
    GaussHermiteQuadrature(num_points::Int)


"""
struct GaussHermiteQuadrature <: AbstractIntegrater
    num_points::Int
end

# f should be a function which eats an individual `x` and returns a univariate distribution.
struct UnivariateFactorisedLikelihood{Tf}
    f::Tf
end

(l::UnivariateFactorisedLikelihood)(x::AbstractVector) = Product(map(l.f, x))

function build_reconstruction_term(
    integrater::GaussHermiteQuadrature,
    latent_gp::LatentGP,
    y::AbstractVector,
)
    integrands = build_integrands(latent_gp, y)
    return (m̃, σ̃²) -> sum(batch_quadrature(integrands, m̃, sqrt.(σ̃²), integrater.num_points))
end

function build_integrands(
    latent_gp::LatentGP{<:AbstractGP, <:UnivariateFactorisedLikelihood},
    y::AbstractVector,
)
    lik = latent_gp.lik.f
    return map(y_ -> (f -> logpdf(lik(f), y_)), y)
end


"""

"""
function optimize_elbo(
    build_latent_gp,
    integrater::AbstractIntegrater,
    x::AbstractVector,
    y::AbstractVector,
    θ0::AbstractVector{<:Real},
    optimiser,
    options,
)
    # Initialise variational parameters. We'll be mutating these to enable warm-starts
    # during each outer iteration of the algorithm.
    __η1 = zeros(length(x))
    __η2 = -ones(length(x))

    function objective(θ::AbstractVector{<:Real})

        # Unflatten θ and build the model at the current hyperparameters.
        l = build_latent_gp(θ)
        r = build_reconstruction_term(integrater, l, y)

        # Optimise the approximate posterior. Drop the gradient because we're
        # differentiating through the optimum.
        η1_opt, η2_opt = Zygote.ignore() do
            η1, η2, iters, delta = optimise_approx_posterior(
                l.f, x, __η1, __η2, r, 1; tol=1e-4,
            )
            __η1 .= η1
            __η2 .= η2
            println((iters, delta))
            return η1, η2
        end

        # Compute the negation of the elbo.
        return -elbo(l.f, x, η1_opt, η2_opt, r)
    end

    training_results = Optim.optimize(
        objective, θ -> only(Zygote.gradient(objective, θ)), θ0, optimiser, options;
        inplace=false,
    )

    l = build_latent_gp(training_results.minimizer)
    r = build_reconstruction_term(integrater, l, y)
    η1, η2, iters, delta = optimise_approx_posterior(l.f, x, __η1, __η2, r, 1; tol=1e-4)
    approx_post = approx_posterior(l.f, x, η1, η2)

    results_summary = (
        training_results=training_results,
        η1=__η1,
        η2=__η2,
        iters=iters,
        delta=delta,
    )

    return approx_post, results_summary
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
    mq, σ²q = mean_and_var(approx_posterior(f, x, η1, η2)(x))
    return logpdf(f(x, σ̃²), ỹ) + r(mq, σ²q) - gaussian_reconstruction_term(ỹ, σ̃², mq, σ²q)
end
