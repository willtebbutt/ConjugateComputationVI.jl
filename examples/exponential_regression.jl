using AbstractGPs
using ConjugateComputationVI
using Distributions
using Optim
using ParameterHandling
using Plots
using Random
using RDatasets
using StatsFuns
using Zygote

using ConjugateComputationVI:
    approx_posterior,
    batch_quadrature,
    elbo,
    optimise_approx_posterior

# Specify a model.
θ_init = (
    scale=positive(1.9),
    stretch=positive(0.8),
    β = fixed(0.3),
)

θ_init_flat, unflatten = ParameterHandling.flatten(θ_init)

build_gp(θ::AbstractVector{<:Real}) = build_gp(ParameterHandling.value(unflatten(θ)))
build_gp(θ::NamedTuple) = GP(θ.scale * AbstractGPs.transform(SEKernel(), θ.stretch))

function build_conditionals(θ::NamedTuple, N::Int)
    return fill(f -> Exponential(exp(f)), N)
end

x = range(-5.0, 5.0; length=100);
x_tr = x
θ_init_val = ParameterHandling.value(θ_init)
f = build_gp(θ_init_val)

# Generate some synthetic data.
y = map(
    (f, conditional) -> rand(conditional(f)),
    rand(f(x, 1e-6)),
    build_conditionals(θ_init_val, length(x)),
)
y_tr = y


# Specify reconstruction term.
function make_integrand(y)
    return (f -> logpdf(Exponential(exp(f)), y))
end

# Specify objective function.
objective(θ::AbstractVector{<:Real}) = objective(ParameterHandling.value(unflatten(θ)))
function objective(θ::NamedTuple)

    # Construct the model.
    f = build_gp(θ)
    
    # Construct the reconstruction term.
    integrands = map(make_integrand, y_tr)
    r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(σ̃²), 10))

    # Optimise the approximate posterior. Drop the gradient because we're differentiating
    # through the optimum.
    η1_opt, η2_opt = Zygote.ignore() do
        η1_0 = randn(length(x_tr))
        η2_0 = -rand(length(x_tr)) .- 1
        η1, η2, iters, delta = optimise_approx_posterior(
            f, x_tr, η1_0, η2_0, r, 1 - 1e-3; tol=1e-8, max_iterations=1_000,
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

f = build_gp(training_results.minimizer)

# f = build_gp(θ_init_flat)
integrands = map(make_integrand, y_tr)
r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(σ̃²), 10))

η1_0 = randn(length(x_tr))
η2_0 = -rand(length(x_tr)) .- 1
η1, η2, iters, delta = optimise_approx_posterior(
    f, x_tr, η1_0, η2_0, r, 1 - 1e-6; tol=1e-12,
)

# Make predictions for the observations and the latent function.
function latent_marginals(x::AbstractVector)
    return marginals(approx_posterior(f, x_tr, η1, η2)(x, 1e-6))
end

function approx_post_marginal_samples(x::AbstractVector, N::Int)
    ms = latent_marginals(x)

    # Generate N samples for each element.
    return map(latent_marginals(x)) do latent_marginal
        f = rand(latent_marginal, N)
        return rand.(Exponential.(exp.(f)))
    end
end

function approx_post_95_CI(x::AbstractVector, N::Int)
    samples = approx_post_marginal_samples(x, N)
    return quantile.(samples, Ref((0.025, 0.5, 0.975)))
end

# Plot the predictions.
x_pr = range(-6.0, 6.0; length=250);
qs = approx_post_95_CI(x_pr, 10_000);
p1 = plot(
    x_pr, getindex.(qs, 1);
    linealpha=0,
    fillrange=getindex.(qs, 3),
    label="95% CI",
    fillalpha=0.3,
);
scatter!(p1, x_tr, y_tr; markersize=2, label="Observations");

p2 = plot(approx_posterior(f, x_tr, η1, η2)(x_pr, 1e-6); label="approx posterior latent");
sampleplot!(approx_posterior(f, x_tr, η1, η2)(x_pr, 1e-6), 10);

plot(p1, p2; layout=(2, 1))

# Look at the callibration.
