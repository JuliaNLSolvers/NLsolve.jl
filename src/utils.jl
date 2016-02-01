sumabs2j(S::AbstractMatrix, j::Integer) = sumabs2(slice(S, :, j))

function sumabs2j(S::SparseMatrixCSC, j::Integer)
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

function assess_convergence(x::Vector,
                            x_previous::Vector,
                            f::Vector,
                            xtol::Real,
                            ftol::Real)
    x_converged, f_converged = false, false

    if !any(isnan, x_previous) && chebyshev(x, x_previous) < xtol
        x_converged = true
    end

    if norm(f, Inf) < ftol
        f_converged = true
    end

    converged = x_converged || f_converged

    return x_converged, f_converged, converged
end

function check_isfinite(x::Vector)
    i = find(!isfinite(x))
    if !isempty(i)
        throw(IsFiniteException(i))
    end
end