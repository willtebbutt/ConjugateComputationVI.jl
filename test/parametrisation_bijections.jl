@testset "parametrisation_bijections" begin
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
