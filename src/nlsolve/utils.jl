function wdot{T}(wx::AbstractArray{T}, x::AbstractArray{T},
                 wy::AbstractArray{T}, y::AbstractArray{T})
    out = zero(T)
    @inbounds @simd for i in 1:length(x)
        out += wx[i]*x[i] * wy[i]*y[i]
    end
    return out
end

wnorm{T}(w::AbstractArray{T}, x::AbstractArray{T}) = sqrt(wdot(w, x, w, x))
assess_convergence(f, ftol) = assess_convergence(NaN, NaN, f, NaN, ftol)
function assess_convergence(x,
                            x_previous,
                            f,
                            xtol,
                            ftol)
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

function check_isfinite(x::AbstractArray)
    i = find((!).(isfinite.(x)))
    if !isempty(i)
        throw(IsFiniteException(i))
    end
end
