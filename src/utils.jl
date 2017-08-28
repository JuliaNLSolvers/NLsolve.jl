function wdot{T}(wx::AbstractVector{T}, x::AbstractVector{T},
                 wy::AbstractVector{T}, y::AbstractVector{T})
    out = zero(T)
    @inbounds @simd for i in 1:length(x)
        out += wx[i]*x[i] * wy[i]*y[i]
    end
    return out
end

wnorm{T}(w::AbstractVector{T}, x::AbstractVector{T}) = sqrt(wdot(w, x, w, x))

function assess_convergence(x::AbstractArray,
                            x_previous::AbstractArray,
                            f::AbstractArray,
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
    i = find((!).(isfinite.(x)))
    if !isempty(i)
        throw(IsFiniteException(i))
    end
end

# Helpers for functions that do not modify arguments in place but return
function not_in_place(f::Function)
    function f!(x::AbstractVector, fx::AbstractVector)
        copy!(fx, f(x))
    end
end

function not_in_place(f::Function, initial_x::AbstractArray)
    function fvec!(x::AbstractVector, fx::AbstractVector)
        copy!(reshape(fx, size(initial_x)...), f(reshape(x, size(initial_x)...)))
    end
end
function not_in_place(f::Function, g::Function)
    DifferentiableMultivariateFunction(not_in_place(f), not_in_place_g(g))
end

function not_in_place(f::Function, g::Function, initial_x::AbstractArray)
    DifferentiableMultivariateFunction(not_in_place(f, initial_x),
                                       not_in_place_g(g, initial_x))
end

function not_in_place(f::Function, g::Function, fg::Function)
    DifferentiableMultivariateFunction(not_in_place(f), not_in_place_g(g),
                                       not_in_place_fg(fg))
end

function not_in_place(f::Function, g::Function, fg::Function, initial_x::AbstractArray)
    DifferentiableMultivariateFunction(not_in_place(f, initial_x),
                                       not_in_place_g(g, initial_x),
                                       not_in_place_fg(fg, initial_x))
end

function not_in_place_g(g::Function)
    function g!(x::AbstractVector, gx::AbstractMatrix)
        copy!(gx, g(x))
    end
end

function not_in_place_g(g::Function, initial_x::AbstractArray)
    function g!(x::AbstractVector, gx::AbstractMatrix)
        copy!(gx, g(reshape(x, size(initial_x)...)))
    end
end

function not_in_place_fg(fg::Function)
    function fg!(x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix)
        (fvec, fjac) = fg(x)
        copy!(fx, fvec)
        copy!(gx, fjac)
    end
end

function not_in_place_fg(fg::Function, initial_x::AbstractArray)
    function fg!(x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix)
        (fvec, fjac) = fg(reshape(x, size(initial_x)...))
        copy!(reshape(fx, size(initial_x)...), fvec)
        copy!(gx, fjac)
    end
end

# Helper for functions that take several scalar arguments and return a tuple
function n_ary(f::Function)
    f!(x::Vector, fx::AbstractArray) = copy!(fx, [f(x...)... ])
end

# Helpers for reshaping functions on arbitrary arrays to functions on vectors
function reshape_f(f!::Function, initial_x::AbstractArray)
    function fvec!(x::AbstractVector, fx::AbstractVector)
        f!(reshape(x, size(initial_x)...), reshape(fx, size(initial_x)...))
    end
end

function reshape_g(g!::Function, initial_x::AbstractArray)
    function gvec!(x::AbstractVector, gx::AbstractMatrix)
        g!(reshape(x, size(initial_x)...), gx)
    end
end

function reshape_g_sparse(g!::Function, initial_x::AbstractArray)
    function gvec!(x::AbstractVector, gx::SparseMatrixCSC)
        g!(reshape(x, size(initial_x)...), gx)
    end
end

function reshape_fg(fg!::Function, initial_x::AbstractArray)
    function fgvec!(x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix)
        fg!(reshape(x, size(initial_x)...), reshape(fx, size(initial_x)...), gx)
    end
end

function reshape_fg_sparse(fg!::Function, initial_x::AbstractArray)
    function fgvec!(x::AbstractVector, fx::AbstractVector, gx::SparseMatrixCSC)
        fg!(reshape(x, size(initial_x)...), reshape(fx, size(initial_x)...), gx)
    end
end
