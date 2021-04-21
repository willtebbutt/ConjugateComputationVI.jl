using AbstractGPs
using ConjugateComputationVI
using Distributions
using Optim
using ParameterHandling
using Plots
using Random
using RDatasets
using StatsFuns
using Stheno
using Zygote

using ConjugateComputationVI: GaussHermiteQuadrature, UnivariateFactorisedLikelihood

# Specify a model.
θ_init = (
    f1 = (
        scale=positive(1.9),
        stretch=positive(0.8),
    ),
    f2 = (
        scale=positive(1.0),
        stretch=positive(0.3),
    ),
);

θ_init_flat, unflatten = ParameterHandling.flatten(θ_init);

function build_latent_gp(θ::AbstractVector{<:Real})
    return build_latent_gp(ParameterHandling.value(unflatten(θ)))
end
function build_latent_gp(θ::NamedTuple)
    gp = @gppp let
        f1 = θ.f1.scale * stretch(GP(SEKernel()), θ.f1.stretch)
        f2 = θ.f2.scale * stretch(GP(SEKernel()), θ.f2.stretch)
        f3 = f1 + f2
    end
    lik = UnivariateFactorisedLikelihood(f -> Exponential(exp(f)))
    return LatentGP(gp, lik, 1e-3)
end

# Generate some synthetic data.
x_f2 = GPPPInput(:f2, range(-5.0, 5.0; length=100));
x_f3 = GPPPInput(:f3, range(-5.0, 5.0; length=150));
x = BlockData(x_f2, x_f3);
y = rand(build_latent_gp(θ_init_flat)(x)).y;
y_f2, y_f3 = split(x, y);

# Add some noise to the initialisation to make this more interesting.
f_approx_post, results_summary = ConjugateComputationVI.optimize_elbo(
    build_latent_gp,
    GaussHermiteQuadrature(10),
    x,
    y,
    θ_init_flat + randn(length(θ_init_flat)),
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(
        show_trace = true,
        iterations=50,
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

θ_opt = results_summary.training_results.minimizer;
x_pr_raw = range(-6.0, 6.0; length=250);
x_pr_f1 = GPPPInput(:f1, x_pr_raw);
x_pr_f2 = GPPPInput(:f2, x_pr_raw);
x_pr_f3 = GPPPInput(:f3, x_pr_raw);
x_pr = BlockData(x_pr_f1, x_pr_f2, x_pr_f3);

# Plot the predictions.
_, qs_f2, qs_f3 = split(x_pr, approx_post_95_CI(x_pr, 10_000));
p1 = plot(
    x_pr_raw, getindex.(qs_f2, 1);
    linealpha=0,
    fillrange=getindex.(qs_f2, 3),
    label="95% CI",
    fillalpha=0.3,
);
scatter!(p1, x_f2.x, y_f2; markersize=2, label="Observations f2");

p2 = plot(
    x_pr_raw, getindex.(qs_f3, 1);
    linealpha=0,
    fillrange=getindex.(qs_f3, 3),
    label="95% CI",
    fillalpha=0.3,
);
scatter!(p2, x_f3.x, y_f3; markersize=2, label="Observations of f3")

p3 = plot();
plot!(p3, x_pr_raw, f_approx_post(x_pr_f1, 1e-6); ribbon_scale=3, color=:blue, label="f1");
sampleplot!(p3, x_pr_raw, f_approx_post(x_pr_f1, 1e-6); samples=3, color=:blue);
plot!(p3, x_pr_raw, f_approx_post(x_pr_f2, 1e-6); ribbon_scale=3, color=:red, label="f2");
sampleplot!(p3, x_pr_raw, f_approx_post(x_pr_f2, 1e-6); samples=3, color=:red);
plot!(p3, x_pr_raw, f_approx_post(x_pr_f3, 1e-6); ribbon_scale=3, color=:green, label="f3");
sampleplot!(p3, x_pr_raw, f_approx_post(x_pr_f3, 1e-6); samples=3, color=:green);

plot(p1, p2, p3; layout=(3, 1))
