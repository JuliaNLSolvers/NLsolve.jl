NLsolve.jl
==========

Solving non-linear systems of equations in Julia.

NLsolve.jl is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

[![Build Status](https://travis-ci.org/JuliaNLSolvers/NLsolve.jl.svg?branch=master)](https://travis-ci.org/JuliaNLSolvers/NLsolve.jl)

[![DOI](https://zenodo.org/badge/14562045.svg)](https://zenodo.org/badge/latestdoi/14562045)


# Non-linear systems of equations
The NLsolve package solves systems of nonlinear equations. Formally, if `F` is
a multivalued function, then this package looks for some vector `x` that
satisfies `F(x)=0` to some accuracy.

The package is also able to solve mixed complementarity problems, which are
similar to systems of nonlinear equations, except that the equality to zero is
allowed to become an inequality if some boundary condition is satisfied. See
further below for a formal definition and the related commands.

There is also an identical API for solving fixed points (i.e., taking as input a function `F(x)`, and solving `F(x) = x`).

Note, if a single equation and not a system is to be solved and performance is not critical, consider using [Roots.jl](https://github.com/JuliaMath/Roots.jl).
If you want to solve a small system of equations at high performance, consider using [NonlinearSolve.jl](https://github.com/JuliaComputing/NonlinearSolve.jl).

# A super simple example

We consider the following bivariate function of two variables:

```jl
(x, y) -> [(x+3)*(y^3-7)+18, sin(y*exp(x)-1)]
```

In order to find a zero of this function and display it, you would write the
following program:

```jl
using NLsolve

function f(x)
    [(x[1]+3)*(x[2]^3-7)+18,
    sin(x[2]*exp(x[1])-1)]
end

sol = nlsolve(f, [ 0.1, 1.2])
sol.zero
```
The first argument to `nlsolve` is the function to be solved which
takes a vector as input and returns the residual as a vector.
The second argument is the starting guess for algorithm.
The `sol.zero` retrieves the solution, if converged.

# A simple example

Continuing on the same system of equations, but now using an in-place
function and a user-specified Jacobian for improved performance:

```jl
using NLsolve

function f!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

function j!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end

nlsolve(f!, j!, [ 0.1; 1.2])
```

First, note that the function `f!` computes the residuals of the nonlinear
system, and stores them in a preallocated vector passed as first argument.
Similarly, the function `j!` computes the Jacobian of the system and stores it
in a preallocated matrix passed as first argument. Residuals and Jacobian
functions can take different shapes, see below.

Second, the `nlsolve` function returns an object of type `SolverResults`. In
particular, the field `zero` of that structure contains the solution if
convergence has occurred. If `r` is an object of type `SolverResults`, then
`converged(r)` indicates if convergence has occurred.

# Ways to specify the function and its Jacobian

There are various ways of specifying the residuals function and possibly its
Jacobian.

## With functions modifying arguments in-place

This is the most efficient method, because it minimizes the memory allocations.

In the following, it is assumed that you have defined a function
`f!(F::AbstractVector, x::AbstractVector)` or, more generally,
`f!(F::AbstractArray, x::AbstractArray)` computing the residual of the system at point `x` and putting it into the `F` argument.

In turn, there 3 ways of specifying how the Jacobian should be computed:

### Finite differencing

If you do not have a function that compute the Jacobian, it is possible to
have it computed by finite difference. In that case, the syntax is simply:

```jl
nlsolve(f!, initial_x)
```

Alternatively, you can construct an object of type
`OnceDifferentiable` and pass it to `nlsolve`, as in:

```jl

initial_x = ...
initial_F = similar(initial_x)
df = OnceDifferentiable(f!, initial_x, initial_F)
nlsolve(df, initial_x)
```
Notice, we passed `initial_x` and `initial_F` to the constructor for `df`. This
does not need to be the actual initial `x` and the residual vector at `x`, but it is used to
initialize cache variables in `df`, so the types and dimensions
of them have to be as if they were.

### Automatic differentiation

Another option if you do not have a function computing the Jacobian is to use
automatic differentiation, thanks to the `ForwardDiff` package. The syntax is
simply:

```jl
nlsolve(f!, initial_x, autodiff = :forward)
```

### Jacobian available

If, in addition to `f!(F::AbstractArray, x::AbstractArray)`, you have a function `j!(J::AbstractArray, x::AbstractArray)` for computing the Jacobian of the system, then the syntax is, as in the example above:

```jl
nlsolve(f!, j!, initial_x)
```

Again it is also possible to specify two functions `f!(F::AbstractArray, x::AbstractArray)` and `j!(J::AbstractArray, x::AbstractArray)` that work on arbitrary arrays `x`.

Note, that you should not assume that the Jacobian `J` passed into `j!` is initialized to a zero matrix. You must set all the elements of the matrix in the function `j!`.

Alternatively, you can construct an object of type
`OnceDifferentiable` and pass it to `nlsolve`, as in:

```jl
df = OnceDifferentiable(f!, j!, initial_x, initial_F)
nlsolve(df, initial_x)
```

### Optimization of simultaneous residuals and Jacobian

If, in addition to `f!` and `j!`, you have a function `fj!(F::AbstractArray, J::AbstractArray, x::AbstractArray)` that computes both the residual and the
Jacobian at the same time, you can use the following syntax

```jl
df = OnceDifferentiable(f!, j!, fj!, initial_x, initial_F)
nlsolve(df, initial_x)
```

If the function `fj!` uses some optimization that make it cost less than
calling `f!` and `j!` successively, then this syntax can possibly improve the
performance.

### Providing only fj!

If a function is available for calculating residuals and the Jacobian,
there is a special syntax for an, arguably, simpler approach. First,
define the function as
```jl
function myfun!(F, J, x)
    # shared calculations begin
    # ...
    # shared calculation end
    if !(F == nothing)
        # mutating calculations specific to f! goes here
    end
    if !(J == nothing)
        # mutating calculations specific to j! goes
    end
end
```

and solve using

```jl
nlsolve(only_fj!(myfun), initial_x)
```

This will make enable `nlsolve` to efficiently calculate `F(x)` and `J(x)`
together, but still be efficient when calculating either `F(x)` or `J(x)`
separately.

## With functions returning residuals and Jacobian as output

Here it is assumed that you have a function `f(x::AbstractArray)` that returns
a newly-allocated vector containing the residuals. Simply pass it to `nlsolve`,
and it will automatically detect if `f` is defined for one or two arguments:

```jl
nlsolve(f, initial_x)
```

Note, that this means that if you have a function `f` with a method that accepts
one argument, and another method that accepts two arguments, it will assume that
the two argument version is a mutating `f`, such as described above.

Via the `autodiff` keyword both finite-differencing and autodifferentiation can
be used to compute the Jacobian in that case.

If, in addition to `f(x::AbstractArray)`, there is a function
`j(x::AbstractArray)` returning a newly-allocated matrix containing the
Jacobian, we again simply pass these to `nlsolve`:

```jl
nlsolve(f, j, initial_x)
```

If, in addition to `f` and `j`, there is a function `fj` returning a tuple of a
newly-allocated vector of residuals and a newly-allocated matrix of the
Jacobian, the approach is the same:

```jl
nlsolve(f, j, fj, initial_x)
```

## With functions taking several scalar arguments

If you have a function `f(x::Float64, y::Float64, ...)` that takes the point of
interest as several scalars and returns a vector or a tuple containing the
residuals, you can use the helper function `n_ary`. The complete syntax is
therefore:

```jl
nlsolve(n_ary(f), initial_x)
```

Finite-differencing is used to compute the Jacobian.

## If the Jacobian is sparse

If the Jacobian of your function is sparse, it is possible to ask the routines
to manipulate sparse matrices instead of full ones, in order to increase
performance on large systems. This means that we must necessarily provide an
appropriate Jacobian type so the solver knows what to feed `j!`.

```jl
df = OnceDifferentiable(f!, j!, x0, F0, J0)
nlsolve(df, initial_x)
```

It is possible to give an optional third function `fj!` to the constructor, as
for the full Jacobian case.

Note that the Jacobian matrix is not reset across function calls. As a result,
you need to be careful and ensure that you
don't forget to overwrite all nonzeros elements that could have been
initialized by a previous function call. If in doubt, you can clear the sparse
matrix at the beginning of the function. If `J` is the sparse Jacobian, this
can be achieved with:

```jl
fill!(a, 0)
dropzeros!(a) # if you also want to remove the sparsity pattern
```

# Fine tunings

Three algorithms are currently available. The choice between these is achieved
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

This is the classical Newton algorithm with optional linesearch.

This method is selected with `method = :newton`.

This method accepts a custom parameter `linesearch`, which must be equal to a
function computing the linesearch. Currently, available values are taken from
the [`LineSearches`](https://github.com/JuliaNLSolvers/LineSearches.jl) package.
By default, no linesearch is performed.
**Note:** it is assumed that a passed linesearch function will at least update the solution
vector and evaluate the function at the new point.

If `method = :newton` and `linesearch=LineSearches.Static()` (the default), an additional parameter
`apply_step!` (default value `(x, x_old, newton_step)->(x .= x_old .+ newton_step)`) can be used to
define a problem-specific function to update the `x` value after the Newton iteration. 
For example, `apply_step! = (x, x_old, newton_step)->(x .= x_old .+ newton_step; x .= max.(x, 1e-80))` 
will enforce a constraint `x[i] >= 1e-80`.

## Anderson acceleration

This method is selected with `method = :anderson`.

It is also known as DIIS or Pulay mixing, this method is based on the
acceleration of the fixed-point iteration `xₙ₊₁ = xₙ + beta*f(xₙ)`, where
by default `beta=1`. It does not use Jacobian information or linesearch,
but has a history whose size is controlled by the `m` parameter: `m=0`
corresponds to the simple fixed-point iteration above,
and higher values use a larger history size to accelerate the
iterations. Higher values of `m` usually increase the speed of
convergence, but increase the storage and computation requirements and
might lead to instabilities. This method is useful to accelerate a
fixed-point iteration `xₙ₊₁ = g(xₙ)` (in which case use this solver
with `f(x) = g(x) - x`).

Reference: H. Walker, P. Ni, Anderson acceleration for fixed-point
iterations, SIAM Journal on Numerical Analysis, 2011

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
* `extended_trace`: should additifonal algorithm internals be added to the state
  trace? Default: `false`.

## Fixed Points

There is a `fixedpoint()` wrapper around `nlsolve()` which maps an input function `F(x)` to `G(x) = F(x) - x`, and likewise for the in-place. This allows convenient solution of fixed-point problems, e.g. of the kind commonly encountered in computational economics. Some notes:

* The default method is `:anderson` with `m = 5`. Naive "Picard"-style iteration can be achieved by setting `m=0`, but that isn't advisable for contractions whose Lipschitz constants are close to 1. If convergence fails, though, you may consider lowering it.
* Autodifferentiation is supported; e.g. `fixedpoint(f!, init_x; method = :newton, autodiff = :forward)`.
* Tolerances and iteration bounds can be set exactly as in `nlsolve()`, since this function is a wrapper, e.g. `fixedpoint(f, init_x; iterations = 500, ...)`.

**Note:** If you are supplying your own derivative, make sure that it is appropriately transformed (i.e., we currently map `f -> f - x`, but are waiting on the API to stabilize before mapping `J -> J - I`, so you'll need to do that yourself.)

# Mixed complementarity problems

Given a multivariate function `f` and two vectors `a` and `b`, the solution to
the mixed complementarity problem (MCP) is a vector `x` such that one of the
following holds for every index `i`:

* either `f_i(x) = 0` and `a_i <= x_i <= b_i`
* or `f_i(x) > 0` and `x_i = a_i`
* or `f_i(x) < 0` and `x_i = b_i`

The vector `a` can contain elements equal to `-Inf`, while the vector
`b` can contain elements equal to `Inf`. In the particular case where all
elements of `a` are equal to `-Inf`, and all elements of `b` are equal to
`Inf`, the MCP is exactly equivalent to the multivariate root finding problem
described above.

The package solves MCPs by reformulating them as the solution to a system of
nonlinear equations (as described by Miranda and Fackler, 2002, though NLsolve
uses the sign convention opposite to theirs).

The function `mcpsolve` solves MCPs. It takes the same arguments as `nlsolve`,
except that the vectors `a` and `b` must immediately follow the argument(s)
corresponding to `f` (and possibly its derivative). There is also an extra
optional argument `reformulation`, which can take two values:

* `reformulation = :smooth`: use a smooth reformulation of the problem using
  the Fischer function. This is the default, since it is more robust for complex
  problems.
* `reformulation = :minmax`: use a min-max reformulation of the problem. It is
  faster than the smooth approximation, since it uses less algebra, but is less
  robust since the reformulated problem has kinks.

Here is a complete example:

```jl
using NLsolve

function f!(F, x)
    F[1]=3*x[1]^2+2*x[1]*x[2]+2*x[2]^2+x[3]+3*x[4]-6
    F[2]=2*x[1]^2+x[1]+x[2]^2+3*x[3]+2*x[4]-2
    F[3]=3*x[1]^2+x[1]*x[2]+2*x[2]^2+2*x[3]+3*x[4]-1
    F[4]=x[1]^2+3*x[2]^2+2*x[3]+3*x[4]-3
end

r = mcpsolve(f!, [0., 0., 0., 0.], [Inf, Inf, Inf, Inf],
             [1.25, 0., 0., 0.5], reformulation = :smooth, autodiff = :forward)
```

The solution is:

```jl
julia> r.zero
4-element Array{Float64,1}:
  1.22474
  0.0
 -1.378e-19
  0.5
```

The lower bounds are hit for the second and third components, hence the second
and third components of the function are positive at the solution. On the other
hand, the first and fourth components of the function are zero at the solution.

```jl
julia> F = similar(r.zero)

julia> f!(F, r.zero)

julia> F
4-element Array{Float64,1}:
 -1.26298e-9
  3.22474
  5.0
  3.62723e-11
```

# Todolist

* Broyden updating of Jacobian in trust-region
* Homotopy methods
* [LMMCP algorithm by C. Kanzow](http://www.mathematik.uni-wuerzburg.de/~kanzow/)

# Related Packages

* [JuMP.jl](https://github.com/JuliaOpt/JuMP.jl) can also solve non linear equations. Just reformulate your problem as an optimization problem with non linear constraints: use the set of equations as constraints, and enter 1.0 as the objective function. JuMP currently supports a number of open-source and commercial solvers.
* [Complementarity.jl](https://github.com/chkwon/Complementarity.jl) brings the powerful modeling language of JuMP.jl to complementarity problems. It supports two solvers: [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl) and NLsolve.jl.

# References

Nocedal, Jorge and Wright, Stephen J. (2006): "Numerical Optimization", second
edition, Springer

[MINPACK](http://www.netlib.org/minpack/) by Jorge More', Burt Garbow, and Ken
Hillstrom at Argonne National Laboratory

Miranda, Mario J. and Fackler, Paul L. (2002): "Applied Computational Economics
and Finance", MIT Press
