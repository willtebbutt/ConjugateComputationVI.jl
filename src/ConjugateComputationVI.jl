module ConjugateComputationVI

"""
    compute_reconstruction(q, logp)

Compute the so-called reconstruction term given by `E_{q(z)}[logp(y | z)]` for RV `z` and
observations `y`.
"""
function compute_reconstruction_gradient(qs, likelihoods) end

function step_ascent(ŷ, observation_models, prior, posterior_marginals, l)

    # Compute the approximate posterior marginals in terms of their expectation parameters.
    qs = posterior_marginals(prior, observation_models, ŷ)

    # Compute gradient of reconstruction term w.r.t. expectation parameters.
    ∇μ = compute_reconstruction_gradient(qs, likelihoods)

    # Update pseudo-observations.
    ŷ_new = (1 - l) * ŷ + l * ∇μ

    # Return updated parameters and gradients (as they're useful for stopping conditions).
    return ŷ_new, ∇μ
end

# There's still quite a lot to figure out here:
# 1. What is the correct way to represent the object that is the surrogate observations /
#   likelihood? There's kind of two things going on, but you only really want to have to
#   consider one inside this algorithm. Might make sense to have a couple of case studies?
# 2. How can we add gradients to surrogate observations / likelihoods? Should we insist on a
#   flat representation, or is something structured okay? Maybe just require that an
#   `update` function is implemented?
# 3. How will compute_reconstruction_gradient wind up being implemented? Is this the right
#   way to go about it? Should we just assume that the user will want to use Zygote?


end
