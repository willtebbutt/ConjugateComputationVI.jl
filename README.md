# ConjugateComputationVI

[![Build Status](https://github.com/willtebbutt/ConjugateComputationVI.jl/workflows/CI/badge.svg)](https://github.com/willtebbutt/ConjugateComputationVI.jl/actions)
[![Coverage](https://codecov.io/gh/willtebbutt/ConjugateComputationVI.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/willtebbutt/ConjugateComputationVI.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

This is an implementation of [1].
It utilises the [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) interface, so should play nicely with any `AbstractGP`, including those from [Stheno.jl](https://github.com/JuliaGaussianProcesses/Stheno.jl) and [TemporalGPs.jl](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl).
No attempt has been made to make this implementation work for anything other than Gaussian processes.

This approximations does not presently play nicely with pseudo-point approximations, although one could certainly attempt to find ways to make it do so.

[1] - Khan, Mohammad, and Wu Lin. "Conjugate-computation variational inference: Converting variational inference in non-conjugate models to inferences in conjugate models." Artificial Intelligence and Statistics. PMLR, 2017.
