NLsolve.jl
==========

The NLsolve package solves systems of nonlinear equations. Formally, if `f` is
a multivariate function, then this package looks for some vector `x` that
satisfies `f(x)=0`.

Since there is some overlap between optimizers and nonlinear solvers, this
package borrows some ideas from the
[Optim](https://github.com/JuliaOpt/Optim.jl) package, and depends on it for
linesearch algorithms.

# Simple example

We consider the following bivariate function of two variables:

    (x, y) -> ((x+3)*(y^3-7)+18, sin(y*exp(x)-1))

In order to find a zero of this function and display it, you would write the
following program:

    using NLsolve
    
    function f!(x, fvec)
        fvec[1] = (x[1]+3)*(x[2]^3-7)+18
        fvec[2] = sin(x[2]*exp(x[1])-1)
    end
    
    function g!(x, fjac)
        fjac[1, 1] = x[2]^3-7
        fjac[1, 2] = 3*x[2]^2*(x[1]+3)
        u = exp(x[1])*cos(x[2]*exp(x[1])-1)
        fjac[2, 1] = x[2]*u
        fjac[2, 2] = u
    end

    nlsolve(f!, g!, [ 0.1; 1.2])

First, note that the function `f!` computes the residuals of the nonlinear
system, and stores them in a preallocated vector passed as second argument.
Similarly, the function `g!` computes the Jacobian of the system and stores it
in a preallocated matrix passed as second argument.

Second, when calling the `nlsolve` function, it is necessary to give a starting
point to the iterative algorithm.

Finally, the `nlsolve` function returns an object of type `SolverResults`. In
particular, the field `zero` of that structure contains the solution if
convergence has occurred. If `r` is an object of type `SolverResults`, then
`converged(r)` indicates if convergence has occurred.

# Specifying the function and its Jacobian

In the following, it is assumed that have defined a function `f!(x::Vector,
fx::Vector)` computing the residual of the system.

There are 3 ways of specifying the function to be solved, depending on the
availability of the Jacobian function:

## Finite differencing

If you do not have a function that compute the Jacobian, it is possible to
have it computed by finite difference. In that case, the syntax is simply:

    nlsolve(f!, initial_x)

Alternatively, you can construct an object of type
`DifferentiableMultivariateFunction` and pass it to `nlsolve`, as in:

    df = DifferentiableMultivariateFunction(f!)
    nlsolve(df, initial_x)

## Jacobian available

If, in addition to `f!`, you have a function `g!(x::Vector, gx::Array)` for
computing the Jacobian of the system, then the syntax is, as in the example
above:

    nlsolve(f!, g!, initial_x)

Alternatively, you can construct an object of type
`DifferentiableMultivariateFunction` and pass it to `nlsolve`, as in:

    df = DifferentiableMultivariateFunction(f!, g!)
    nlsolve(df, initial_x)

## Optimization of simultaneous residuals and Jacobian

If, in addition to `f!` and `g!`, you have a function `fg!(x::Vector,
fx::Vector, gx::Array)` that computes both the residual and the Jacobian at
the same time, you can use the following syntax:

    df = DifferentiableMultivariateFunction(f!, g!, fg!)
    nlsolve(df, initial_x)

If the function `fg!` uses some optimization that make it costless than
calling `f!` and `g!` successively, then this syntax can possibly improve the
performance.

# Fine tunings

Two algorithms are currently available. The choice between the two is achieved
by setting the optional `method` argument of `nlsolve`. The default algorithm
is the trust region method.

## Trust region method

This is the well-known solution method which relies on a quadratic
approximation of the least-squares objective, considered to be valid over a
compact region centered around the current iterate.

This method is selected with `method = :trust_region`.

This method accepts the following custom parameters:

* `factor`: determines the size of the initial trust region. This size is set
  to the product of factor and the euclidean norm of `initial_x` if nonzero, or
  else to factor itself. Default: `1.0`.
* `autoscale`: if `true`, then the variables will be automatically rescaled.
  The scaling factors are the norms of the Jacobian columns. Default: `true`.

## Newton method with linesearch

This is the classical Newton algorithm with linesearch.

This method is selected with `method = :newton`.

This method accepts a custom parameter `linesearch!`, which must be equal to a
function computing the linesearch. Currently, available values are taken from
the `Optim` package, and are: `Optim.backtracking_linesearch!` (the default),
`Optim.hz_linesearch!`, `Optim.interpolating_linesearch!`.

## Common options

Other optional arguments to `nlsolve`, available for all algorithms, are:

* `xtol`: norm difference in `x` between two successive iterates under which
  convergence is declared. Default: `0.0`.
* `ftol`: infinite norm of residuals under which convergence is declared.
  Default: `1e-8`.
* `iterations`: maximum number of iterations. Default: `1_000`.
* `store_trace`: should a trace of the optimization algorithm's state be
  stored? Default: `false`.
* `show_trace`: should a trace of the optimization algorithm's state be shown
  on `STDOUT`? Default: `false`.
* `extended_trace`: should additional algorithm internals be added to the state
  trace? Default: `false`.

# Todolist

* Broyden updating of Jacobian in trust-region
* Macro to run tests against all algorithms
* Homotopy methods

# References

Nocedal, Jorge and Wright, Stephen J. (2006): "Numerical Optimization", second
edition, Springer

[MINPACK](http://www.netlib.org/minpack/) by Jorge More', Burt Garbow, and Ken
Hillstrom at Argonne National Laboratory
