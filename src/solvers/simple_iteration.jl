# Simple Iteration

function simple_iteration{T}(df::OnceDifferentiable, initial_x::AbstractArray{T}, xtol::Real, ftol::Real, iterations::Integer)
  f0 = df.f(initial_x) #Jasmine: this is how you call f
  residual = Inf
  iter = 1
  xold = initial_x

  #TODO: This is assuming out of place.  Would need to check and do a special csae if inplace operations.

  while residual > ftol && iter < iterations
      xnew = df.f(xold) #You call the embedded f function like this
      residual = norm(xold - xnew);
      xold = xnew
      iter += 1
  end

  return (xold,iter)
end
