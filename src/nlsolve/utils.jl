wdot(wx, x, wy, y) = dot(wx.*x, wy.*y)
wnorm(w, x) = norm(w.*x)

assess_convergence(f, ftol) = assess_convergence(NaN, NaN, f, NaN, ftol)
function assess_convergence(x,
                            x_previous,
                            f,
                            xtol,
                            ftol)
    x_converged, f_converged = false, false
    if norm(x-x_previous) <= xtol
        x_converged = true
    end
    if maximum(abs, f) <= ftol
        f_converged = true
    end

    return x_converged, f_converged
end

function check_isfinite(x::AbstractArray)
    if any(!isfinite, x)
        i = findall(!isfinite, x)
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
