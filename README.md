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
convergence has occurred.

# Fine tunings

Two algorithms are currently available. The choice between the two is achieved
by setting the optional `method` argument of `nlsolve`. The default algorithm
is the trust region method.

## Trust region method

This is the well-known solution method which relies on a quadratic
approximation of the least-squares objective, considered to be valid over a
compact region centered around the current iterate.

This method is selected with `method = :trust_region`.

This method accepts a custom parameter `factor`, which determines the size of
the initial trust region. This size is set to the product of factor and the
euclidean norm of `initial_x` if nonzero, or else to factor itself. The default
value is `1.0`.

## Newton method with linesearch

This is the classical Newton algorithm with linesearch.

This method is selected with `method = :newton`.

This method accepts a custom parameter `lineasearch!`, which must be equal to a
function computing the linesearch. Currently, available values are taken from
the `Optim` package, and are: `Optim.hz_linesearch!` (the default),
`Optim.backtracking_linesearch!`, `Optim.interpolating_linesearch!`.

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

* Jacobian by finite difference
* Broyden updating of Jacobian in trust-region
* Autoscaling in trust-region
* Add more elaborate tests
* Macro to run tests against all algorithms

# References

Nocedal, Jorge and Wright, Stephen J. (2006): "Numerical Optimization", second
edition, Springer
