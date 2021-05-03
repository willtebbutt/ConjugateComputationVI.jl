module ConjugateComputationVI

using AbstractGPs
using ChainRulesCore
using FastGaussQuadrature
using LinearAlgebra
using Optim
using Zygote

using AbstractGPs: AbstractGP, LatentFiniteGP, Normal

include("quadrature.jl")
include("parametrisation_bijections.jl")
include("approximate_inference.jl")

end
