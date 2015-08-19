sumabs2j(S::AbstractMatrix, j::Integer) = sumabs2(slice(S, :, j))

if VERSION < v"0.4.0-dev+1307" 
    nzrange(S::Base.SparseMatrixCSC, col::Integer) = S.colptr[col]:(S.colptr[col+1]-1)
end

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