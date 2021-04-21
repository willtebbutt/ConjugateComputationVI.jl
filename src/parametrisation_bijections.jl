"""
    natural_from_canonical(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})

Compute natural parameters `η1` and `η2` from canonical paramters `y` and `σ²`.
Inverse of [`canonical_from_natural`](@ref).
"""
function natural_from_canonical(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    return y ./ σ², (-1) ./ (2 .* σ²)
end

"""
    canonical_from_natural(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})

Compute canonical parameters `y` and `σ²` from natural parameters `η1` and `η2`.
Inverse of [`natural_from_canonical`](@ref).
"""
function canonical_from_natural(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})
    σ² = -1 ./ (2 .* η2)
    y = σ² .* η1
    return y, σ²
end

"""
    expectation_from_canonical(m::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})

Compute expectation parameters `m1` and `m2` from canonical parameters `m` and `σ²`.
Inverse of [`expectation_from_canonical`](@ref).
"""
function expectation_from_canonical(m::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    return m, m.^2 .+ σ²
end

"""
    canonical_from_expectation(m1::AbstractVector{<:Real}, m2::AbstractVector{<:Real})

Compute canonical parameters `m` and `σ²` from expectation parameters `m1` and `m2`.
Inverse of [`expectation_from_canonical`](@ref).
"""
function canonical_from_expectation(m1::AbstractVector{<:Real}, m2::AbstractVector{<:Real})
    return m1, m2 .- m1.^2
end
