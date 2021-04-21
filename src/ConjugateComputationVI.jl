module ConjugateComputationVI

using AbstractGPs
using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Optim
using Zygote

using AbstractGPs: AbstractGP, LatentFiniteGP

include("quadrature.jl")
include("parametrisation_bijections.jl")
include("approximate_inference.jl")

end
