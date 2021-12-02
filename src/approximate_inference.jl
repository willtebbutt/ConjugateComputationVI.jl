struct SimpleNormal{Tm, Tσ}
    m::Tm
    σ::Tσ
end

function AbstractGPs.logpdf(d::SimpleNormal, y::Real)
    m = Zygote.literal_getfield(d, Val(:m))
    σ = Zygote.literal_getfield(d, Val(:σ))
    return -(log(2π) + 2log(σ) + ((y - m) / σ)^2) / 2
end

function ChainRulesCore.rrule(::typeof(AbstractGPs.mean), d::T) where {T<:Normal}
    function mean_pullback(Δ::Real)
        return NoTangent, Tangent{T}(μ=Δ)
    end
    return d.μ, mean_pullback
end

function ChainRulesCore.rrule(::typeof(AbstractGPs.var), d::T) where {T<:Normal}
    function var_pullback(Δ::Real)
        NoTangent, Tangent{T}(σ=2Δ * d.σ)
    end
    return d.σ^2, var_pullback
end

"""
    gaussian_reconstruction_term(
        y::AbstractVector{<:Real},
        σ²::AbstractVector{<:Real},
        m̃::AbstractVector{<:Real},
        σ̃²::AbstractVector{<:Real},
    )

Computes
```julia
∫ logpdf(Normal(f, σ²[n]), y[n]) pdf(Normal(m̃[n], σ̃²[n]), f) df
```
for each `n ∈ eachindex(y)`.
"""
function gaussian_reconstruction_term(
    y::AbstractVector{<:Real},
    Σy::Diagonal{<:Real},
    mq::AbstractVector{<:Real},
    Σq::Diagonal{<:Real},
)
    return sum(
        map(
            (y, σ², m̃, σ̃²) -> logpdf(SimpleNormal(m̃, sqrt(σ²)), y) - σ̃² / (2σ²),
            y, diag(Σy), mq, diag(Σq),
        ),
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
    η2::Diagonal{<:Real},
)
    y, Σ = canonical_from_natural(η1, η2)
    return posterior(f(x, Σ + 1e-6 * I), y)
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
    η2::Diagonal{<:Real},
    r,
    ρ::Real,
)
    # Check that the step size makes sense.
    (ρ > 1 || ρ <= 0) && throw(error("Bad step size"))

    # Compute the approximate posterior marginals.
    mq, σ²q = mean_and_var(approx_posterior(f, x, η1, η2)(x))

    # Compute both of the expectation parameters.
    m1, m2 = expectation_from_canonical(mq, Diagonal(σ²q))

    # Compute the gradient w.r.t. both of the expectation parameters. This is equivalent to
    # the natural gradient w.r.t. the natural parameters.
    g1, g2 = Zygote.gradient(
        (m1, m2) -> begin
            y, σ² = canonical_from_expectation(m1, m2)
            return r(y, σ²)
        end,
        m1, m2,
    )

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
    η2::Diagonal{<:Real},
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

Compute the evidence lower bound associated with the GP `f`, with reconstruction term `r`,
variational parameters `η1` and `η2`, and collection of inputs `x`.
"""
function AbstractGPs.elbo(
    f::AbstractGP,
    x::AbstractVector,
    η1::AbstractVector{<:Real},
    η2::Diagonal{<:Real},
    r,
)
    ỹ, σ̃² = canonical_from_natural(η1, η2)
    mq, σ²q = mean_and_var(approx_posterior(f, x, η1, η2)(x))
    return logpdf(f(x, σ̃²), ỹ) + r(mq, Diagonal(σ²q)) - gaussian_reconstruction_term(ỹ, σ̃², mq, Diagonal(σ²q))
end



abstract type AbstractIntegrater end

"""
    GaussHermiteQuadrature(num_points::Int)

Specify that the expectation required for the reconstruction term is to be computed using
Gauss-Hermite quadrature with `num_points` quadrature points. 
"""
struct GaussHermiteQuadrature <: AbstractIntegrater
    num_points::Int
end

"""
    UnivariateFactorisedLikelihood(build_lik)

A likelihood function which factorises across inputs.
Applying this likelihood to any vector `f::AbstractVector{<:Real}` yields a `Product`
distribution.
Exposing this structure is useful when working with quadrature methods.
"""
struct UnivariateFactorisedLikelihood{Tf}
    build_lik::Tf
end

function (l::UnivariateFactorisedLikelihood)(x::AbstractVector{<:Real})
    return AbstractGPs.Product(map(l.build_lik, x))
end

function build_reconstruction_term(
    integrater::GaussHermiteQuadrature,
    latent_gp::LatentGP,
    y::AbstractVector,
)
    integrands = build_integrands(latent_gp, y)
    function reconstruction_term(mq::AbstractVector{<:Real}, Σq::Diagonal{<:Real})
        return sum(batch_quadrature(integrands, mq, map(sqrt, diag(Σq)), integrater.num_points))
    end
    return reconstruction_term
end

function build_integrands(
    latent_gp::LatentGP{<:AbstractGP, <:UnivariateFactorisedLikelihood},
    y::AbstractVector,
)
    lik = latent_gp.lik.build_lik
    return map(y_ -> (f -> logpdf(lik(f), y_)), y)
end

"""
    optimize_elbo(
        build_latent_gp,
        integrater::AbstractIntegrater,
        x::AbstractVector,
        y::AbstractVector,
        θ0::AbstractVector{<:Real},
        optimiser,
        options,
    )

Optimise the elbo w.r.t. model parameters.

# Arguments
- `build_latent_gp`: a unary function accepting an `AbstractVector{<:Real}` returning a
    LatentGP
- `integrater`::AbstractIntegrater: an object specifying how to perform inference
- `x::AbstractVector`: a collection of inputs
- `y::AbstractVector`: a collection of outputs
- `θ0::AbstractVector{<:Real}`: initial model parameter values
- `optimiser`: an optimiser that can be passed to `Optim.optimize`
- `options`: options to be passed to `Optim.optimize`

# Returns
- `AbstractGP`: the approximate posterior at the optimum
- `NamedTuple`: results summary containing various useful bits of information
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
    __η2 = Diagonal(-ones(length(x)))

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
