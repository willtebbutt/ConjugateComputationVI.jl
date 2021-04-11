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

# Get the (infamous) Iris dataset, and only keep two labels -- we'll just do binary
# classification for now.
iris = filter(
    row -> row.Species == "setosa" || row.Species == "virginica",
    dataset("datasets", "iris"),
)

setosa = filter(row -> row.Species == "setosa", iris)
virginica = filter(row -> row.Species == "virginica", iris)

scatter(setosa.SepalLength, setosa.SepalWidth; label="Setosa")
scatter!(virginica.SepalLength, virginica.SepalWidth; label="Virginica")


# Check that there are only two features.
@assert length(unique(iris.Species)) == 2

# Encode the specifies using Ints. Could also have used Booleans.
y = map(x -> x == "setosa" ? 1 : 0, iris.Species)

# Construct input matrix from feature columns.
X = hcat(iris.SepalLength, iris.SepalWidth)
x = ColVecs(collect(X'))

# Split into train and test sets.
train_indices = randperm(length(y))[1:50]
test_indices = setdiff(eachindex(y), train_indices)
x_tr = x[train_indices]
x_te = x[test_indices]
y_tr = y[train_indices]
y_te = y[test_indices]

θ_init = (
    scale=fixed(1.0),
    stretch=positive.(ones(2)),
)

θ_init_flat, unflatten = ParameterHandling.flatten(θ_init)

build_gp(θ::AbstractVector{<:Real}) = build_gp(ParameterHandling.value(unflatten(θ)))
build_gp(θ::NamedTuple) = GP(θ.scale * AbstractGPs.transform(SEKernel(), θ.stretch))

# Specify reconstruction term.
function make_integrand(y)
    return f -> logpdf(Bernoulli(logistic(f)), y)
end
integrands = map(make_integrand, y_tr)

# Specify the reconstruction term.
r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(σ̃²), 10))

function objective(θ::AbstractVector{<:Real})

    # Unflatten θ and build the model at the current hyperparameters.
    f = build_gp(θ)

    # Optimise the approximate posterior. Drop the gradient because we're differentiating
    # through the optimum.
    η1_opt, η2_opt = Zygote.ignore() do
        η1_0 = randn(length(x_tr))
        η2_0 = -rand(length(x_tr)) .- 1
        η1, η2, iters, delta = optimise_approx_posterior(
            f, x_tr, η1_0, η2_0, r, 1 - 1e-6; tol=1e-8, max_iterations=1_000,
        )
        println((iters, delta))
        return η1, η2
    end

    # Compute the negation of the elbo.
    return -elbo(f, x_tr, η1_opt, η2_opt, r)
end

objective(θ_init_flat)
Zygote.gradient(objective, θ_init_flat)

# Optimise the ELBO using Optim.
training_results = Optim.optimize(
    objective,
    θ -> only(Zygote.gradient(objective, θ)),
    θ_init_flat,
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
η1_0 = randn(length(x_tr))
η2_0 = -rand(length(x_tr)) .- 1
η1, η2, iters, delta = optimise_approx_posterior(
    f, x_tr, η1_0, η2_0, r, 1 - 1e-6; tol=1e-12,
)

function latent_marginals(x::AbstractVector)
    return marginals(approx_posterior(f, x_tr, η1, η2)(x))
end

function post_mean_prediction(x::AbstractVector)
    ms = latent_marginals(x)
    conditionals = fill(f -> pdf(Bernoulli(logistic(f)), 1), length(ms))
    posterior_mean_predictions = batch_quadrature(conditionals, mean.(ms), std.(ms), 25)
    return posterior_mean_predictions
end

# Compute predictions using the optimal hyperparameters.
function MAP_predict(x::AbstractVector)
    posterior_mean_predictions = post_mean_prediction(x)
    MAP_predictions = map(x -> x > 0.5 ? 1 : 0, posterior_mean_predictions)
    return MAP_predictions
end

# Compute training set accuracy as a %age.
100 * mean(MAP_predict(x_tr) .== y_tr)

# Compute test set accuracy as a %age.
100 * mean(MAP_predict(x_te) .== y_te)

# Compute prediction surface.
x1_range = range(minimum(X[:, 1]) - 3, maximum(X[:, 1]) + 3; length=100);
x2_range = range(minimum(X[:, 2]) - 3, maximum(X[:, 2]) + 3; length=100);
x_pr_vecs = [[x1, x2] for x1 in x1_range, x2 in x2_range];
X_pr = reduce(hcat, x_pr_vecs);
x_pr = ColVecs(X_pr);


# Plot classification.
MAP_pr = MAP_predict(x_pr);
p1 = heatmap(x1_range, x2_range, reshape(MAP_pr, length(x2_range), length(x1_range))'; label="classifier");
scatter!(p1, setosa.SepalLength, setosa.SepalWidth; label="");
scatter!(p1, virginica.SepalLength, virginica.SepalWidth; label="");

# Plot posterior mean prob.
post_mean_pr = post_mean_prediction(x_pr);
p2 = heatmap(x1_range, x2_range, reshape(post_mean_pr, length(x2_range), length(x1_range))'; label="prob");
scatter!(p2, setosa.SepalLength, setosa.SepalWidth; label="");
scatter!(p2, virginica.SepalLength, virginica.SepalWidth; label="");

# Plot posterior marginals.
ms = latent_marginals(x_pr);
p3 = heatmap(x1_range, x2_range, reshape(mean.(ms), length(x2_range), length(x1_range))'; label="latent mean");
scatter!(p3, setosa.SepalLength, setosa.SepalWidth; label="");
scatter!(p3, virginica.SepalLength, virginica.SepalWidth; label="");

p4 = heatmap(x1_range, x2_range, reshape(std.(ms), length(x2_range), length(x1_range))'; label="latent std");
scatter!(p4, setosa.SepalLength, setosa.SepalWidth; label="");
scatter!(p4, virginica.SepalLength, virginica.SepalWidth; label="");

plot(p1, p2, p3, p4; layout=(2, 2))
