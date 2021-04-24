using AbstractGPs
using ConjugateComputationVI
using Distributions
using Optim
using ParameterHandling
using Plots
using Random
using StatsBase
using StatsFuns
using TemporalGPs
using Zygote

using ConjugateComputationVI: GaussHermiteQuadrature, UnivariateFactorisedLikelihood

# Adjoint for the Poisson logpdf.
# Someone should implement this in the correct place.
# log(λ^x exp(-λ) / x!) =
# x log(λ) - λ - log(x!)
# dλ = x / λ - 1
Zygote.@adjoint function StatsFuns.poislogpdf(λ::Float64, x::Union{Float64, Int})
    function poislogpdf_pullback(Δ::Real)
        return Δ * (x / λ - 1), nothing
    end
    return StatsFuns.poislogpdf(λ, x), poislogpdf_pullback
end

function Zygote._pullback(ctx::Zygote.AContext, ::typeof(logpdf), d::Poisson, x)
    out, pb = Zygote._pullback(ctx, poislogpdf, d.λ, x)
    function logpdf_pullback(Δ::Real)
        _, Δλ, Δx = pb(Δ)
        return nothing, (λ=Δλ, ), Δx
    end
    return out, logpdf_pullback
end

# Specify a model.
θ_init = (scale=positive(1.0), stretch=positive(1.0));
θ_init_flat, unflatten = ParameterHandling.flatten(θ_init);

function build_latent_gp(θ::AbstractVector{<:Real})
    return build_latent_gp(ParameterHandling.value(unflatten(θ)))
end
function build_latent_gp(θ::NamedTuple)
    gp_naive = GP(θ.scale * AbstractGPs.transform(Matern52Kernel(), θ.stretch))
    gp = to_sde(gp_naive, SArrayStorage(Float64))
    lik = UnivariateFactorisedLikelihood(f -> Poisson{Float64}(exp(f)))
    return LatentGP(gp, lik, 1e-9)
end

# Generate synthetic data.
x = RegularSpacing(0.0, 0.001, 10_000);
y = rand(build_latent_gp(θ_init_flat)(x)).y;

f_approx_post, results_summary = ConjugateComputationVI.optimize_elbo(
    build_latent_gp,
    GaussHermiteQuadrature(10),
    x,
    y,
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
        return exp.(f)
    end
end

function approx_post_95_CI(x::AbstractVector, N::Int)
    samples = approx_post_marginal_samples(x, N)
    return quantile.(samples, Ref((0.025, 0.5, 0.975)))
end

# Plot the predictions.
qs = approx_post_95_CI(x, 100);
p1 = plot(
    x, getindex.(qs, 1);
    linealpha=0,
    fillrange=getindex.(qs, 3),
    label="95% CI",
    fillalpha=0.3,
    color=:blue,
);
scatter!(p1, x, y; markersize=1, label="Observations");

p2 = plot(f_approx_post(x, 1e-6); ribbon_scale=3, color=:blue, label="approx post latent");

plot(p1, p2; layout=(2, 1))
