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

        @testset "ad" begin
            out, pb = Zygote.pullback(canonical_from_expectation, m1, m2)

            # Ensure that forwards-pass isn't modified.
            @test out[1] ≈ y_recovered
            @test out[2] ≈ σ²_recovered

            # Compute reverse-pass using Zygote and FiniteDifferences.
            Δout = (randn(length(out[1])), randn(length(out[2])))

            Δm1, Δm2 = pb(Δout)
            Δm1_fd, Δm2_fd = FiniteDifferences.j′vp(
                central_fdm(5, 1), canonical_from_expectation, Δout, m1, m2,
            )
            @test Δm1 ≈ Δm1_fd
            @test Δm2 ≈ Δm2_fd
        end
    end
end
