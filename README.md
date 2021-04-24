# ConjugateComputationVI

[![Build Status](https://github.com/willtebbutt/ConjugateComputationVI.jl/workflows/CI/badge.svg)](https://github.com/willtebbutt/ConjugateComputationVI.jl/actions)
[![Coverage](https://codecov.io/gh/willtebbutt/ConjugateComputationVI.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/willtebbutt/ConjugateComputationVI.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

This is an implementation of [1].
It utilises the [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) interface, so should play nicely with any `AbstractGP`, including those from [Stheno.jl](https://github.com/JuliaGaussianProcesses/Stheno.jl) and [TemporalGPs.jl](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl).
No attempt has been made to make this implementation work for anything other than Gaussian processes.



# Example Usage

Approximate inference and learning in a GP under an Exponential likelihood.
This is primarily handled using the `build_latent_gp` function, which produces a `LatentGP`
specifying this model when provided with kernel parameters.
[ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl/) is used to handle
the book-keeping associated with the model parameters.

```julia
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

using ConjugateComputationVI: GaussHermiteQuadrature, UnivariateFactorisedLikelihood

# Specify the model parameters.
θ_init = (scale=positive(1.9), stretch=positive(0.8));
θ_init_flat, unflatten = ParameterHandling.flatten(θ_init);

# Specify the model.
# A core requirement of this package is that you are able to provide a function mapping
# from your model parameters to a `LatentGP`.
function build_latent_gp(θ::AbstractVector{<:Real})
    return build_latent_gp(ParameterHandling.value(unflatten(θ)))
end
function build_latent_gp(θ::NamedTuple)
    gp = GP(θ.scale * AbstractGPs.transform(SEKernel(), θ.stretch))
    lik = UnivariateFactorisedLikelihood(f -> Exponential(exp(f)))
    return LatentGP(gp, lik, 1e-9)
end

# Specify inputs and generate some synthetic outputs.
x = range(-5.0, 5.0; length=100);
y = rand(build_latent_gp(θ_init_flat)(x)).y;

# Attempt to recover the kernel parameters used when generating the data.
# Add some noise to the initialisation to make this more interesting.
# We specify that the reconstruction term in the ELBO is to be approximated using
# Gauss-Hermite quadrature with 10 points.
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
        iterations=25,
        f_calls_limit=50,
    ),
);

# Compute approx. posterior CIs using Monte Carlo.
function approx_post_95_CI(x::AbstractVector, N::Int)
    samples = map(marginals(f_approx_post(x, 1e-6))) do latent_marginal
        f = rand(latent_marginal, N)
        return rand.(Exponential.(exp.(f)))
    end
    return quantile.(samples, Ref((0.025, 0.5, 0.975)))
end

x_pr = range(-6.0, 6.0; length=250);
qs = approx_post_95_CI(x_pr, 10_000);

# Plot the predictions.
p1 = plot(
    x_pr, getindex.(qs, 1);
    linealpha=0,
    fillrange=getindex.(qs, 3),
    label="95% CI",
    fillalpha=0.3,
);
scatter!(p1, x, y; markersize=2, label="Observations");

p2 = plot(
    f_approx_post(x_pr, 1e-6);
    ribbon_scale=3, color=:blue, label="approx posterior latent",
);
sampleplot!(f_approx_post(x_pr, 1e-6), 10; color=:blue);

plot(p1, p2; layout=(2, 1))
```

See the examples directory for more.

# Limitations

This approximation does not presently play nicely with pseudo-point approximations.
That would be an interesting research direction.



# References

[1] - Khan, Mohammad, and Wu Lin. "Conjugate-computation variational inference: Converting variational inference in non-conjugate models to inferences in conjugate models." Artificial Intelligence and Statistics. PMLR, 2017.
