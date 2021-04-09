using AbstractGPs
using ConjugateComputationVI
using KernelFunctions
using TemporalGPs
using Test

using ConjugateComputationVI:
    approx_posterior,
    canonical_pseudo_obs,
    natural_pseudo_obs

@testset "ConjugateComputationVI.jl" begin
    @testset "parameter_conversion" begin
        y = randn(10)
        σ² = randn(10)
        η1, η2 = natural_pseudo_obs(y, σ²)
        y_recovered, σ²_recovered = canonical_pseudo_obs(η1, η2)

        # Check that the transformations are at least each others inverse.
        @test y_recovered ≈ y
        @test σ² ≈ σ²_recovered
    end
    @testset "approx_posterior" begin

        # Generate a GP and sample from it.
        f = GP(SEKernel())
        x = collect(range(-5.0, 5.0; length=31))
        σ² = rand(31) .+ 1e-2
        y = rand(f(x, σ²))

        # Compute natural pseudo-observations and approximate posterior.
        η1, η2 = natural_pseudo_obs(y, σ²)
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
end
