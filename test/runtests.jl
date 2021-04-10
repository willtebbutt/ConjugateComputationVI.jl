using AbstractGPs
using ConjugateComputationVI
using Distributions
using KernelFunctions
using Random
using TemporalGPs
using Test
using Zygote

using ConjugateComputationVI:
    approx_posterior,
    canonical_from_expectation,
    canonical_from_natural,
    expectation_from_canonical,
    natural_from_canonical,
    optimise_approx_posterior,
    update_approx_posterior

function generate_synthetic_problem(rng::AbstractRNG)
    f = GP(Matern52Kernel())
    x = collect(range(-5.0, 5.0; length=31))
    σ² = rand(rng, 31) .+ 1e-2
    y = rand(rng, f(x, σ²))
    return f, x, σ², y
end

@testset "ConjugateComputationVI.jl" begin
    @testset "parameter_conversion" begin
        _, _, σ², y = generate_synthetic_problem(MersenneTwister(123456))

        @testset "natural parameter conversion" begin
            η1, η2 = natural_from_canonical(y, σ²)
            y_recovered, σ²_recovered = canonical_from_natural(η1, η2)

            # Check that the transformations are at least each others inverse.
            @test y_recovered ≈ y
            @test σ² ≈ σ²_recovered
        end
        @testset "expectation parameter conversion" begin
            m1, m2 = expectation_from_canonical(y, σ²)
            y_recovered, σ²_recovered = canonical_from_expectation(m1, m2)

            # Check that the transformations are at least each others inverse.
            @test y_recovered ≈ y
            @test σ² ≈ σ²_recovered
        end
    end
    @testset "approx_posterior" begin
        f, x, σ², y = generate_synthetic_problem(MersenneTwister(123456))

        # Compute natural pseudo-observations and approximate posterior.
        η1, η2 = natural_from_canonical(y, σ²)
        f_post_approx = approx_posterior(f, x, η1, η2)

        # Compute the exact posterior.
        f_post_exact = posterior(f(x, σ²), y)

        # The exact and approximate posteriors should be identical in this case.
        # Check that this is the case by comparing the statistics at some test points.
        x_test = range(-7.5, 7.5; length=101)
        ms_exact = marginals(f_post_exact(x_test))
        ms_approx = marginals(f_post_approx(x_test))
        @test mean.(ms_exact) ≈ mean.(ms_approx)
        @test std.(ms_exact) ≈ std.(ms_approx)
    end
    @testset "update_approx_posterior" begin
        f, x, σ², y = generate_synthetic_problem(MersenneTwister(123456))

        # Reconstruction term and its gradient for Gaussian likelihood.
        r(m̃, σ̃²) = sum(logpdf.(Normal.(m̃, sqrt.(σ²)), y) .- σ̃² ./ (2 .* σ²))
        ∇r = (m̃, σ̃²) -> Zygote.gradient(r, m̃, σ̃²)

        @testset "optimal parameters don't change when updated" begin

            # Run a step of the update procedure.
            η1, η2 = natural_from_canonical(y, σ²)
            η1_new, η2_new, g1, g2 = update_approx_posterior(f, x, η1, η2, ∇r, 1.0)

            # Verify that the new parameters are equal to the old parameters.
            @test η1 ≈ η1_new
            @test η2 ≈ η2_new
            @test g1 ≈ η1_new
            @test g2 ≈ η2_new
        end

        # Verify the optimal approximate posterior parameters are found after only a single
        # iteration of the algorithm in the Gaussian case.
        @testset "optimal in 1 step" begin

            # Perform a single update step with unit step size.
            η1, η2 = natural_from_canonical(y .+ randn(length(y)), σ² .+ rand(length(y)))
            η1_opt, η2_opt, g1, g2 = update_approx_posterior(f, x, η1, η2, ∇r, 1.0)

            # Ensure that we've reached the optimum (the exact posterior).
            y_opt, σ²_opt = canonical_from_natural(η1_opt, η2_opt)
            @test y ≈ y_opt
            @test σ² ≈ σ²_opt
            @test g1 ≈ η1_opt
            @test g2 ≈ η2_opt
        end
    end
    @testset "optimise_approx_posterior" begin
        f, x, σ², y = generate_synthetic_problem(MersenneTwister(123456))

        # Reconstruction term and its gradient for Gaussian likelihood.
        r(m̃, σ̃²) = sum(logpdf.(Normal.(m̃, sqrt.(σ²)), y) .- σ̃² ./ (2 .* σ²))
        ∇r = (m̃, σ̃²) -> Zygote.gradient(r, m̃, σ̃²)

        @testset "converges quickly for step size $ρ" for ρ in [0.1, 0.5, 0.9, 1.0]
            η1, η2 = natural_from_canonical(y, σ²)
            η1_0 = η1 + randn(length(η1))
            η2_0 = η2 - rand(length(y))
            η1_opt, η2_opt, _, _ = optimise_approx_posterior(f, x, η1_0, η2_0, ∇r, ρ)
            @test η1_opt ≈ η1
            @test η2_opt ≈ η2
        end
    end
end
