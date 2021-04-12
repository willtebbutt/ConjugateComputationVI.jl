using AbstractGPs
using ConjugateComputationVI
using Distributions
using Optim
using ParameterHandling
using Plots
using Random
using RDatasets
using StatsBase
using StatsFuns
using Zygote

using ConjugateComputationVI:
    approx_posterior,
    batch_quadrature,
    elbo,
    optimise_approx_posterior

# Adjoint for the Poisson logpdf.
# log(λ^x exp(-λ) / x!) =
# x log(λ) - λ - log(x!)
# dλ = x / λ - 1
Zygote.@adjoint function StatsFuns.poislogpdf(λ::Float64, x::Union{Float64, Int})
    function poislogpdf_pullback(Δ::Real)
        return Δ * (x / λ - 1), nothing
    end
    return StatsFuns.poislogpdf(λ, x), poislogpdf_pullback
end

# # Check correctness.
# mypoislogpdf(λ, x) = x * log(λ) - λ - log(factorial(x))
# logpdf(Poisson(3.0), 2)
# mypoislogpdf(3.0, 2)
# Zygote.gradient(mypoislogpdf, 3.0, 2)
# Zygote.gradient((λ, x) -> logpdf(Poisson(λ), x), 3.0, 2)

# Download the data.
data = dataset("boot", "coal")

# Bin the data
h = fit(Histogram, data.Date, range(minimum(data.Date), maximum(data.Date); length=200))

# Construct data set to learn on.
x = collect(only(h.edges)[1:end-1])
y = h.weights

# Specify a model.
θ_init = (
    scale=positive(1.0),
    stretch=positive(1e-3),
)

θ_init_flat, unflatten = ParameterHandling.flatten(θ_init)

build_gp(θ::AbstractVector{<:Real}) = build_gp(ParameterHandling.value(unflatten(θ)))
build_gp(θ::NamedTuple) = GP(θ.scale * AbstractGPs.transform(Matern52Kernel(), θ.stretch))

# Specify reconstruction term.
function make_integrand(y)
    return (f -> logpdf(Poisson(exp(f)), y))
end

# Specify objective function.
objective(θ::AbstractVector{<:Real}) = objective(ParameterHandling.value(unflatten(θ)))
function objective(θ::NamedTuple)

    # Construct the model.
    f = build_gp(θ)
    
    # Construct the reconstruction term.
    integrands = map(make_integrand, y)
    r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(σ̃²), 15))

    # Optimise the approximate posterior. Drop the gradient because we're differentiating
    # through the optimum.
    η1_opt, η2_opt = Zygote.ignore() do
        η1_0 = zeros(length(x))
        η2_0 = -ones(length(x))
        η1, η2, iters, delta = optimise_approx_posterior(
            f, x, η1_0, η2_0, r, 1 - 1e-9; tol=1e-12, max_iterations=50,
        )
        println((iters, delta))
        return η1, η2
    end

    # Compute the negative elbo.
    return -elbo(f, x, η1_opt, η2_opt, r)
end

objective(θ_init_flat)

Zygote.gradient(objective, θ_init_flat)

# Learn from a different initialisation.
training_results = Optim.optimize(
    objective,
    θ -> only(Zygote.gradient(objective, θ)),
    θ_init_flat + randn(length(θ_init_flat)),
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(
        show_trace = true,
        iterations=25,
        f_calls_limit=50,
    );
    inplace=false,
)

θ_opt = ParameterHandling.value(unflatten(training_results.minimizer))
f = build_gp(θ_opt)

# f = build_gp(θ_init_flat)
integrands = map(make_integrand, y)
r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(σ̃²), 10))

η1_0 = zeros(length(x));
η2_0 = -ones(length(x));
η1, η2, iters, delta = optimise_approx_posterior(
    f, x, η1_0, η2_0, r, 1 - 1e-3; tol=1e-12,
)

# Make predictions for the observations and the latent function.
function latent_marginals(x::AbstractVector)
    return marginals(approx_posterior(f, x, η1, η2)(x, 1e-6))
end

function approx_post_marginal_samples(x::AbstractVector, N::Int)
    ms = latent_marginals(x)

    # Generate N samples for each element.
    return map(latent_marginals(x)) do latent_marginal
        f = rand(latent_marginal, N)
        return exp.(f) / (x[2] - x[1])
    end
end

function approx_post_95_CI(x::AbstractVector, N::Int)
    samples = approx_post_marginal_samples(x, N)
    return quantile.(samples, Ref((0.025, 0.5, 0.975)))
end

# Plot the predictions.
qs = approx_post_95_CI(x, 10_000);
p1 = plot(
    x, getindex.(qs, 1);
    linealpha=0,
    fillrange=getindex.(qs, 3),
    label="95% CI",
    fillalpha=0.3,
);
scatter!(p1, x, y; markersize=2, label="Observations");

p2 = plot(approx_posterior(f, x, η1, η2)(x, 1e-6); label="approx posterior latent");
sampleplot!(approx_posterior(f, x, η1, η2)(x, 1e-6), 10);

plot(p1, p2; layout=(2, 1))

# Look at the callibration.
