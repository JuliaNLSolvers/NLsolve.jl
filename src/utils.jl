sumabs2j(S::AbstractMatrix, j::Integer) = sumabs2(slice(S, :, j))

function sumabs2j(S::Base.SparseMatrix.SparseMatrixCSC, j::Integer)
    sumabs2(sub(nonzeros(S), nzrange(S, j)))
end

function wdot{T}(w::Vector{T}, x::Vector{T}, y::Vector{T})
    out = zero(T)
    @inbounds @simd for i in 1:length(x)
        out += w[i] * x[i] * y[i]
    end
    return out
end

wnorm{T}(w::Vector{T}, x::Vector{T}) = sqrt(wdot(w, x, x))