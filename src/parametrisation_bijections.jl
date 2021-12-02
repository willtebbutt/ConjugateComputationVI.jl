"""
    natural_from_canonical(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})

Compute natural parameters `η1` and `η2` from canonical paramters `y` and `σ²`.
Inverse of [`canonical_from_natural`](@ref).
"""
function natural_from_canonical(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    η1, η2 = natural_from_canonical(y, Diagonal(σ²))
    return η1, η2.diag
end

# function natural_from_canonical(y::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
#     C = cholesky(Σ)
#     return C \ y, -inv(C) / 2
# end

function natural_from_canonical(y::AbstractVector{<:Real}, Σ::Diagonal{<:Real})
    return Σ \ y, -inv(Σ) / 2
end

"""
    canonical_from_natural(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})

Compute canonical parameters `y` and `σ²` from natural parameters `η1` and `η2`.
Inverse of [`natural_from_canonical`](@ref).
"""
function canonical_from_natural(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})
    y, Σ = canonical_from_natural(η1, Diagonal(η2))
    return y, diag(Σ)
end

function canonical_from_natural(η1::AbstractVector{<:Real}, η2::Diagonal{<:Real})
    Σ = -inv(η2) / 2
    y = Σ * η1
    return y, Σ
end

"""
    expectation_from_canonical(m::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})

Compute expectation parameters `m1` and `m2` from canonical parameters `m` and `σ²`.
Inverse of [`expectation_from_canonical`](@ref).
"""
function expectation_from_canonical(m::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    m, S = expectation_from_canonical(m, Diagonal(σ²))
    return m, diag(S)
end

function expectation_from_canonical(m::AbstractVector{<:Real}, Σ::Diagonal{<:Real})
    return m, Diagonal(m.^2 + diag(Σ))
end

"""
    canonical_from_expectation(m1::AbstractVector{<:Real}, m2::AbstractVector{<:Real})

Compute canonical parameters `m` and `σ²` from expectation parameters `m1` and `m2`.
Inverse of [`expectation_from_canonical`](@ref).
"""
function canonical_from_expectation(m1::AbstractVector{<:Real}, m2::AbstractVector{<:Real})
    m1, M2 = canonical_from_expectation(m1, Diagonal(m2))
    return m1, diag(M2)
end

function canonical_from_expectation(m1::AbstractVector{<:Real}, M2::Diagonal{<:Real})
    return m1, Diagonal(diag(M2) .- m1.^2)
end

# function Zygote._pullback(
#     ::Zygote.AContext,
#     ::typeof(canonical_from_expectation),
#     m1::AbstractVector{<:Real},
#     m2::AbstractVector{<:Real},
# )
#     function canonical_from_expectation_pullback(Δ)
#         Δ === nothing && return nothing
#         Δm = Δ[1]
#         Δσ² = Δ[2]

#         Δm1 = Δm .- 2 .* Δσ² .* m1
#         Δm2 = Δσ²
#         return (nothing, Δm1, Δm2)
#     end
#     return canonical_from_expectation(m1, m2), canonical_from_expectation_pullback
# end
