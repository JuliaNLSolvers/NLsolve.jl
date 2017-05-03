function wdot{T}(wx::AbstractVector{T}, x::AbstractVector{T},
                 wy::AbstractVector{T}, y::AbstractVector{T})
    out = zero(T)
    @inbounds @simd for i in 1:length(x)
        out += wx[i]*x[i] * wy[i]*y[i]
    end
    return out
end

wnorm{T}(w::AbstractVector{T}, x::AbstractVector{T}) = sqrt(wdot(w, x, w, x))

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
    i = find((!).(@compat isfinite.(x)))
    if !isempty(i)
        throw(IsFiniteException(i))
    end
end

# Helpers for functions that do not modify arguments in place but return
function not_in_place(f::Function)
    f!(x::Vector, y::AbstractArray) = copy!(y, f(x))
end

function not_in_place(f::Function, g::Function)
    DifferentiableMultivariateFunction(not_in_place(f), not_in_place(g))
end

function not_in_place(f::Function, g::Function, fg::Function)
    function fg!(x::Vector, fx::Vector, gx::Array)
        (fvec, fjac) = fg(x)
        copy!(fx, fvec)
        copy!(gx, fjac)
    end
    DifferentiableMultivariateFunction(not_in_place(f), not_in_place(g), fg!)
end

# Helper for functions that take several scalar arguments and return a tuple
function n_ary(f::Function)
    f!(x::Vector, fx::AbstractArray) = copy!(fx, [f(x...)... ])
end
