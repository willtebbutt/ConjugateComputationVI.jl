"""
    natural_from_canonical(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
"""
function natural_from_canonical(y::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    return y ./ σ², (-1) ./ (2 .* σ²)
end

"""
    canonical_from_natural(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})
"""
function canonical_from_natural(η1::AbstractVector{<:Real}, η2::AbstractVector{<:Real})
    σ² = -1 ./ (2 .* η2)
    y = σ² .* η1
    return y, σ²
end

"""
    expectation_from_canonical(m::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
"""
function expectation_from_canonical(m::AbstractVector{<:Real}, σ²::AbstractVector{<:Real})
    return m, m.^2 .+ σ²
end

"""
    canonical_from_expectation(m1::AbstractVector{<:Real}, m2::AbstractVector{<:Real})
"""
function canonical_from_expectation(m1::AbstractVector{<:Real}, m2::AbstractVector{<:Real})
    return m1, m2 .- m1.^2
end
