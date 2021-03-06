using AbstractGPs
using ConjugateComputationVI
using Distributions
using FiniteDifferences
using KernelFunctions
using LinearAlgebra
using Random
using StatsFuns
# using TemporalGPs
using Test
using Zygote

using ConjugateComputationVI:
    approx_posterior,
    batch_quadrature,
    canonical_from_expectation,
    canonical_from_natural,
    expectation_from_canonical,
    gaussian_reconstruction_term,
    natural_from_canonical,
    optimise_approx_posterior,
    update_approx_posterior

function generate_synthetic_problem(rng::AbstractRNG)
    f = GP(Matern52Kernel())
    x = collect(range(-2.0, 2.0; length=7))
    Σ = Diagonal(rand(rng, 7) .+ 1e-2)
    y = rand(rng, f(x, Σ))
    return f, x, Σ, y
end

@testset "ConjugateComputationVI.jl" begin
    include("quadrature.jl")
    include("parametrisation_bijections.jl")
    include("approximate_inference.jl")
end
