module ConjugateComputationVI

using AbstractGPs
using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Zygote

using AbstractGPs: AbstractGP

include("quadrature.jl")
include("parametrisation_bijections.jl")
include("approximate_inference.jl")

end
