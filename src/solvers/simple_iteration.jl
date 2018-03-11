# simple iteration
struct simple_iteration
end

macro simple_iterationtrace(stepnorm)
    esc(quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(cache.x)
            end
            update!(tr,
                    it,
                    maximum(abs, value(df)),
                    $stepnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end)
end

function simple_iteration{T}(df::OnceDifferentiable,
                             initial_x::AbstractArray{T},
                             xtol::Real,
                             ftol::Real,
                             iterations::Integer,
                             store_trace::Bool,
                             show_trace::Bool,
                             extended_trace::Bool)

  f0 = df.f(initial_x) #how to call f
  residual = Inf
  iter = 1
  xold = initial_x

  #TODO:
  copy!(cache.x, initial_x)
  x_converged, f_converged, converged = assess_convergence(value(df), ftol)

  tr = SolverTrace()
  tracing = store_trace || show_trace || extended_trace
  @simple_iterationtrace convert(T, NaN)

  #This is assuming out of place.  Would need to check and do a special csae if inplace operations.

  while residual > ftol && iter < iterations
      xnew = df.f(xold) #You call the embedded f function like this
      residual = norm(xold - xnew);
      xold = xnew
      iter += 1
  end

  #return (xold,iter)
  return SolverResults("Simple Iteration",xold,iter,x_converged, xtol, f_converged,ftol,first(df.f_calls))
end

function simple_iteration{T}(df::OnceDifferentiable,
                   initial_x::AbstractArray{T},
                   xtol::Real,
                   ftol::Real,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool)
    simple_iteration_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace)
end
