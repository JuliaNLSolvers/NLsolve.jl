function wdot(wx::AbstractArray{T}, x::AbstractArray{T},
                 wy::AbstractArray{T}, y::AbstractArray{T}) where T
    out = zero(T)
    @inbounds @simd for i in 1:length(x)
        out += wx[i]*x[i] * wy[i]*y[i]
    end
    return out
end

wnorm(w, x) = sqrt(wdot(w, x, w, x))
assess_convergence(f, f_tol) = assess_convergence(NaN, NaN, f, NaN, f_tol)
function assess_convergence(x,
                            x_previous,
                            f,
                            x_tol,
                            f_tol)
    x_converged, f_converged = false, false

    if !any(isnan, x_previous) && chebyshev(x, x_previous) <= x_tol
        x_converged = true
    end

    if maximum(abs, f) <= f_tol
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

"""
    qrdelete!(Q, R, k)

Delete the left-most column of F = Q[:, 1:k] * R[1:k, 1:k] by updating Q and R.

Only Q[:, 1:(k-1)] and R[1:(k-1), 1:(k-1)] are valid on exit.
"""
function qrdelete!(Q::AbstractMatrix, R::AbstractMatrix, k::Int)
  n, m = size(Q)
  m == LinearAlgebra.checksquare(R) || throw(DimensionMismatch())
  1 ≤ k ≤ m || throw(ArgumentError())

  # apply Givens rotations
  for i in 2:k
      g = first(givens(R, i - 1, i, i))
      lmul!(g, R)
      rmul!(Q, g')
  end

  # move columns of R
  @inbounds for j in 1:(k-1)
    for i in 1:(k-1)
      R[i, j] = R[i, j + 1]
    end
  end

  Q, R
end

"""
    qradd!(Q, R, v, k)

Replace the right-most column of F = Q[:, 1:k] * R[1:k, 1:k] with v by updating Q and R.

This implementation modifies vector v as well. Only Q[:, 1:k] and R[1:k, 1:k] are valid on
exit.
"""
function qradd!(Q::AbstractMatrix, R::AbstractMatrix, v::AbstractVector, k::Int)
  n, m = size(Q)
  n == length(v) || throw(DimensionMismatch())
  m == LinearAlgebra.checksquare(R) || throw(DimensionMismatch())
  1 ≤ k ≤ m || throw(ArgumentError())

  @inbounds for i in 1:(k-1)
    q = view(Q, :, i)
    r = dot(q, v)

    R[i, k] = r
    axpy!(-r, q, v)
  end

  @inbounds begin
    d = norm(v)
    R[k, k] = d
    @. Q[:, k] = v / d
  end

  Q, R
end
