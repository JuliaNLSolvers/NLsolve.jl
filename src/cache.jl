# For simplicity we use the same cache for trust_region and newton.
immutable NLsolveCache{T, Jac}
    v1::Vector{T}
    v2::Vector{T}
    v3::Vector{T}
    v4::Vector{T}
    v5::Vector{T}
    v6::Vector{T}
    v7::Vector{T}
    fjac::Jac
    len::Int
end

NLsolveCache{T}(df::AbstractDifferentiableMultivariateFunction, x::Vector{T}) = NLsolveCache(df, T, length(x))

function NLsolveCache{T}(df::AbstractDifferentiableMultivariateFunction, ::Type{T}, len::Int)
    v1 = Vector{T}(len)
    v2 = Vector{T}(len)
    v3 = Vector{T}(len)
    v4 = Vector{T}(len)
    v5 = Vector{T}(len)
    v6 = Vector{T}(len)
    v7 = Vector{T}(len)

    fjac = alloc_jacobian(df, T, len)

    return NLsolveCache{T, typeof(fjac)}(v1, v2, v3, v4, v5, v6, v7, fjac, len)
end

function check_lengths(cache::NLSolveCache, len::Int)
    if cache.len != len
        throw(ArgumentError("mismatch between cache and initial guess"))
    end
end