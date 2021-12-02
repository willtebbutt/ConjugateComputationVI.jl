@testset "approximate_inference" begin
    @testset "gaussian_reconstruction_term" begin
        _, _, Σ, y = generate_synthetic_problem(MersenneTwister(123456))
        mq = randn(size(y))
        Σq = Diagonal(rand(length(y)) .+ 1)
        gaussian_reconstruction_term(y, Σ, mq, Σq)
        Zygote.gradient(gaussian_reconstruction_term, y, Σ, mq, Σq)
    end
    @testset "approx_posterior" begin
        f, x, Σ, y = generate_synthetic_problem(MersenneTwister(123456))

        # Compute natural pseudo-observations and approximate posterior.
        η1, η2 = natural_from_canonical(y, Σ)
        f_post_approx = approx_posterior(f, x, η1, η2)

        # Compute the exact posterior.
        f_post_exact = posterior(f(x, Σ), y)

        # The exact and approximate posteriors should be identical in this case.
        # Check that this is the case by comparing the statistics at some test points.
        x_test = range(-7.5, 7.5; length=101)
        ms_exact = marginals(f_post_exact(x_test))
        ms_approx = marginals(f_post_approx(x_test))
        @test mean.(ms_exact) ≈ mean.(ms_approx) rtol=1e-6
        @test std.(ms_exact) ≈ std.(ms_approx) rtol=1e-6
    end
    @testset "update_approx_posterior" begin
        f, x, Σ, y = generate_synthetic_problem(MersenneTwister(123456))

        # Reconstruction term and its gradient for Gaussian likelihood.
        r(m̃, σ̃²) = gaussian_reconstruction_term(y, Σ, m̃, σ̃²)

        @testset "optimal parameters don't change when updated" begin

            # Run a step of the update procedure.
            η1, η2 = natural_from_canonical(y, Σ)
            η1_new, η2_new, delta_norm = update_approx_posterior(f, x, η1, η2, r, 1.0)

            # Verify that the new parameters are equal to the old parameters.
            @test η1 ≈ η1_new
            @test η2 ≈ η2_new
            @test delta_norm < 1e-12
        end

        # Verify the optimal approximate posterior parameters are found after only a single
        # iteration of the algorithm in the Gaussian case.
        @testset "optimal in 1 step" begin

            # Perform a single update step with unit step size.
            N = length(y)
            η1, η2 = natural_from_canonical(y .+ randn(N), Σ + Diagonal(rand(N)))
            η1_opt, η2_opt, _ = update_approx_posterior(f, x, η1, η2, r, 1.0)

            # Ensure that we've reached the optimum (the exact posterior).
            y_opt, Σ_opt = canonical_from_natural(η1_opt, η2_opt)
            @test y ≈ y_opt
            @test Σ ≈ Σ_opt
        end
    end
    @testset "optimise_approx_posterior" begin
        f, x, Σ, y = generate_synthetic_problem(MersenneTwister(123456))

        # Reconstruction term and its gradient for Gaussian likelihood.
        r(m̃, σ̃²) = gaussian_reconstruction_term(y, Σ, m̃, σ̃²)

        @testset "converges quickly for step size $ρ" for ρ in [0.1, 0.5, 0.9, 1.0]
            η1, η2 = natural_from_canonical(y, Σ)
            η1_0 = η1 + randn(length(η1))
            η2_0 = η2 - Diagonal(rand(length(y)))
            η1_opt, η2_opt, _ = optimise_approx_posterior(f, x, η1_0, η2_0, r, ρ)
            @test η1_opt ≈ η1
            @test η2_opt ≈ η2
        end
    end
    @testset "elbo" begin
        f, x, Σ, y = generate_synthetic_problem(MersenneTwister(123456))

        # Set initial parameters to be sub-optimal.
        η1, η2 = natural_from_canonical(y, Σ)
        η1_0 = η1 + randn(length(η1))
        η2_0 = η2 - Diagonal(rand(length(y)))

        # Specify the reconstruction term.
        r(m̃, σ̃²) = gaussian_reconstruction_term(y, Σ, m̃, σ̃²)

        # Verify that the ELBO is less than the log marginal likelihood for a Gaussian.
        @test elbo(f, x, η1_0, η2_0, r) < logpdf(f(x, Σ), y)

        # Verify that the ELBO equals the log marginal likelihood at the optimum.
        @test elbo(f, x, η1, η2, r) ≈ logpdf(f(x, Σ), y)

        # Verify that the gradient w.r.t. everything can be computed using Zygote.
        Zygote.gradient(elbo, f, x, η1, η2, r)
    end
    @testset "batch_quadrature" begin
        f, x, Σ, y = generate_synthetic_problem(MersenneTwister(123456))
        function make_integrand(y, σ²)
            return f -> logpdf(Normal(f, sqrt(σ²)), y)
        end
        integrands = map(make_integrand, y, diag(Σ))

        # Specify the reconstruction term.
        r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(diag(σ̃²)), 25))

        # Ensure that it's close to the exact reconstruction term in the Gaussian case.
        mq = randn(length(y))
        Σq = Diagonal(rand(length(y)) .+ 1)
        @test r(mq, Σq) ≈ gaussian_reconstruction_term(y, Σ, mq, Σq)

        # Check that the gradients agree.
        Δmq_exact, ΔΣq_exact = Zygote.gradient(
            (m̃, σ̃²) -> gaussian_reconstruction_term(y, Σ, m̃, σ̃²), mq, Σq,
        )
        Δmq_quad, ΔΣq_quad = Zygote.gradient(r, mq, Σq)
        @test Δmq_exact ≈ Δmq_quad
        @test ΔΣq_exact ≈ ΔΣq_quad
    end
    @testset "Gaussian likelihood via Quadature" begin
        f, x, Σ, y = generate_synthetic_problem(MersenneTwister(123456))
        function make_integrand(y, σ²)
            return f -> logpdf(Normal(f, sqrt(σ²)), y)
        end
        integrands = map(make_integrand, y, diag(Σ))

        # Set initial parameters to be sub-optimal.
        η1, η2 = natural_from_canonical(y, Σ)
        η1_0 = η1 + randn(length(η1))
        η2_0 = η2 - Diagonal(rand(length(y)))

        # Specify the reconstruction term.
        r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(diag(σ̃²)), 25))

        # Ensure that it's close to the exact reconstruction term in the Gaussian case.
        mq = randn(length(y))
        Σq = Diagonal(rand(length(y)) .+ 1)
        @test r(mq, Σq) ≈ gaussian_reconstruction_term(y, Σ, mq, Σq)

        @testset "converges quickly for step size $ρ" for ρ in [0.1, 0.5, 0.9, 1.0]
            η1, η2 = natural_from_canonical(y, Σ)
            η1_0 = η1 + randn(length(η1))
            η2_0 = η2 - Diagonal(rand(length(y)))
            η1_opt, η2_opt, _, _ = optimise_approx_posterior(f, x, η1_0, η2_0, r, ρ)
            @test η1_opt ≈ η1
            @test η2_opt ≈ η2
        end
        @testset "elbo" begin
            # Verify that the ELBO is less than the log marginal likelihood for a Gaussian.
            @test elbo(f, x, η1_0, η2_0, r) < logpdf(f(x, Σ), y)

            # Verify that the ELBO equals the log marginal likelihood at the optimum.
            @test elbo(f, x, η1, η2, r) ≈ logpdf(f(x, Σ), y)

            # Verify that the gradient w.r.t. everything can be computed using Zygote.
            Zygote.gradient(elbo, f, x, η1, η2, r)
        end
    end
    @testset "Bernoulli likelihood via Quadrature" begin
        # Really all that we can do here is ensure that it converges and that the ELBO
        # is _something_.
        f, x, _, _ = generate_synthetic_problem(MersenneTwister(123456))
        y = rand.(Bernoulli.(logistic.(rand(f(x, 1e-4)))))
        function make_integrand(y)
            return f -> logpdf(Bernoulli(logistic(f)), y)
        end
        integrands = map(make_integrand, y)

        # Specify the reconstruction term.
        r(m̃, σ̃²) = sum(batch_quadrature(integrands, m̃, sqrt.(diag(σ̃²)), 25))

        # Set initial parameters to be... something.
        η1_0 = randn(length(y))
        η2_0 = Diagonal(- rand(length(y)))

        @testset "solutions converge to the same thing" begin
            solutions = map([0.1, 0.5, 0.9, 0.95, 0.99]) do ρ
                η1_opt, η2_opt, iteration, delta_norm = optimise_approx_posterior(
                    f, x, η1_0, η2_0, r, ρ,
                )
                return η1_opt, η2_opt
            end
            ref_solutions = first(solutions)
            @test all(isapprox.(first.(solutions), Ref(ref_solutions[1]); rtol=1e-6))
            @test all(isapprox.(last.(solutions), Ref(ref_solutions[2]); rtol=1e-6))
        end
        @testset "elbo" begin

            # Check that we can compute the ELBO. There's nothing obvious to compare it to.
            elbo(f, x, η1_0, η2_0, r)

            # Verify that the gradient w.r.t. everything can be computed using Zygote.
            Zygote.gradient(elbo, f, x, η1_0, η2_0, r)
        end
    end
end
