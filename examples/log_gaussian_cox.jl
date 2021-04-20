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

# Download the data.
data = dataset("boot", "coal")

# Bin the data
h = fit(Histogram, data.Date, range(minimum(data.Date), maximum(data.Date); length=200))

# Construct data set to learn on.
x = collect(only(h.edges)[1:end-1])
y = h.weights

# Specify a model.
θ_init = (scale=positive(1.0), stretch=positive(1e-3))
θ_init_flat, unflatten = ParameterHandling.flatten(θ_init)

build_gp(θ::AbstractVector{<:Real}) = build_gp(ParameterHandling.value(unflatten(θ)))
build_gp(θ::NamedTuple) = GP(θ.scale * AbstractGPs.transform(Matern52Kernel(), θ.stretch))

# Specify reconstruction term.
const integrands_ = map(y_ -> (f -> logpdf(Poisson(exp(f)), y_)), y)
r(m̃, σ̃²) = sum(batch_quadrature(integrands_, m̃, sqrt.(σ̃²), 15))

f_approx_post, results_summary = ConjugateComputationVI.optimize_elbo(
    build_gp,
    x,
    r,
    θ_init_flat,
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(
        show_trace = true,
        iterations=25,
        f_calls_limit=50,
    ),
);

function approx_post_marginal_samples(x::AbstractVector, N::Int)
    return map(marginals(f_approx_post(x))) do latent_marginal
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
    color=:blue,
);
scatter!(p1, x, y; markersize=2, label="Observations");

p2 = plot(f_approx_post(x, 1e-6); ribbon_scale=3, color=:blue, label="approx post latent");
sampleplot!(f_approx_post(x, 1e-6), 10; color=:blue);

plot(p1, p2; layout=(2, 1))
