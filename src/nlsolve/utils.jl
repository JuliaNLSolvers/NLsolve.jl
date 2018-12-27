function wdot(wx::AbstractArray{T}, x::AbstractArray{T},
                 wy::AbstractArray{T}, y::AbstractArray{T}) where T
    out = zero(T)
    @inbounds @simd for i in 1:length(x)
        out += wx[i]*x[i] * wy[i]*y[i]
    end
    return out
end

wnorm(w, x) = sqrt(wdot(w, x, w, x))
assess_convergence(f, ftol) = assess_convergence(NaN, NaN, f, NaN, ftol)
function assess_convergence(x,
                            x_previous,
                            f,
                            xtol,
                            ftol)
    x_converged, f_converged = false, false

    if !any(isnan, x_previous) && chebyshev(x, x_previous) <= xtol
        x_converged = true
    end

    if maximum(abs, f) <= ftol
        f_converged = true
    end

    converged = x_converged || f_converged

    return x_converged, f_converged, converged
end

function check_isfinite(x::AbstractArray)
    if any((y)->!isfinite(y),x)
        i = findall((!).(isfinite.(x)))
        throw(IsFiniteException(i))
    end
end
