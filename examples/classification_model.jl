using AbstractGPs
using ConjugateComputationVI
using RDatasets

using ConjugateComputationVI: approx_posterior, batch_quadrature

# Get the (infamous) Iris dataset, and only keep two labels -- we'll just do binary
# classification for now.
iris = filter(
    row -> row.Species == "setosa" || row.Species == "virginica",
    dataset("datasets", "iris"),
)

# Check that there are only two features.
@assert length(unique(iris.Species)) == 2

# Encode the specifies using Ints. Could also have used Booleans.
y = map(x -> x == "setosa" ? 1 : 0, iris.Species)

# Construct input matrix from feature columns.
x = RowVecs(hcat(iris.SepalLength, iris.SepalWidth, iris.PetalLength, iris.PetalWidth))

# Specify the GP (we'll not do hyperparameter optimisation in the first-pass).
# We'll use a unit length scale. Can always tweak by hand if necessary.
f = GP(SEKernel())
σ² = 1e-3

# Specify reconstruction term.
function make_integrand(y)
    return f -> logpdf(Bernoulli(logistic(f)), y)
end
integrands = map(make_integrand, y)

# Specify the reconstruction term.
r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(σ̃²), 25))

# Run VI to convergence. This should happen in only a few iterations.
η1_0 = randn(length(y))
η2_0 = - rand(length(y))
ρ = 0.9
η1_opt, η2_opt, iteration, delta_norm = optimise_approx_posterior(
    f, x, η1_0, η2_0, r, ρ; max_iterations=5,
)
@show iteration, delta_norm

# Make some predictions using the same batch quadrature routine.
ms = marginals(approx_posterior(f, x, η1_opt, η2_opt)(x))
conditionals = fill(f -> pdf(Bernoulli(logistic(f)), 1), length(ms))
posterior_mean_predictions = batch_quadrature(conditionals, mean.(ms), std.(ms), 25)

# Make MAP predictions and compute %age correctly classified.
MAP_predictions = map(x -> x > 0.5 ? 1 : 0, posterior_mean_predictions)
100 * count(MAP_predictions .== y) / length(y)

# As expected, it is extremely easy to fit the training data in this task.
