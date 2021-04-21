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
    GaussHermiteQuadrature,
    UnivariateFactorisedLikelihood

# Specify a model.
θ_init = (scale=positive(1.9), stretch=positive(0.8), β = fixed(0.3))

θ_init_flat, unflatten = ParameterHandling.flatten(θ_init)

x = range(-5.0, 5.0; length=100);
x_tr = x
θ_init_val = ParameterHandling.value(θ_init)

function build_latent_gp(θ::AbstractVector{<:Real})
    return build_latent_gp(ParameterHandling.value(unflatten(θ)))
end
function build_latent_gp(θ::NamedTuple)
    gp = GP(θ.scale * AbstractGPs.transform(SEKernel(), θ.stretch))
    lik = UnivariateFactorisedLikelihood(f -> Exponential(exp(f)))
    return LatentGP(gp, lik, 1e-9)
end

# Generate some synthetic data.
y_tr = rand(build_latent_gp(θ_init_val)(x_tr)).y;

# Add some noise to the initialisation to make this more interesting.
f_approx_post, results_summary = ConjugateComputationVI.optimize_elbo(
    build_latent_gp,
    GaussHermiteQuadrature(10),
    x_tr,
    y_tr,
    θ_init_flat + randn(length(θ_init_flat)),
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
    return map(marginals(f_approx_post(x, 1e-6))) do latent_marginal
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

p2 = plot(
    f_approx_post(x_pr, 1e-6);
    ribbon_scale=3, color=:blue, label="approx posterior latent",
);
sampleplot!(f_approx_post(x_pr, 1e-6), 10; color=:blue);

plot(p1, p2; layout=(2, 1))
