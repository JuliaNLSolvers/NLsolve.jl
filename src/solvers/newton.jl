struct Newton{LS, LSOL} <: AbstractSolver
    linesearch!::LS
    linsolve!::LSOL
end
"""
# Newton
## Constructor
```julia
Newton(; linesearch = LineSearches.Static(),
linsolve = (x, A, b) -> copy!(x, A\b))
```
## Description
The `Newton` method implements Newton's method for solving nonlinear systems of
equations. See Nocedal and Wright for a discussion of Newton's method in practice.

## Example
```julia
# Pkg.add("IterativeSolvers")

julia> using IterativeSolvers

julia> Newton(linesearch = LineSearches.BackTracking(), linsolve = gmres!)
Newton{LineSearches.BackTracking{Float64,Int64},IterativeSolvers.#gmres!}(LineSearches.BackTracking{Float64,Int64}
  c_1: Float64 0.0001
  ρ_hi: Float64 0.5
  ρ_lo: Float64 0.1
  iterations: Int64 1000
  order: Int64 3
  maxstep: Float64 Inf
, IterativeSolvers.gmres!)
```
## References
- Nocedal, J. and S. J. Wright (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
Newton(; linesearch = LineSearches.Static(), linsolve = (x, A, b) -> copy!(x, A\b)) =
    Newton(linesearch, linsolve)

struct NewtonCache{Tx} <: AbstractSolverCache
    x::Tx
    xold::Tx
    p::Tx
    g::Tx
end
function NewtonCache(df)
    x = similar(df.x_f)
    xold = similar(x)
    p = similar(x)
    g = similar(x)
    NewtonCache(x, xold, p, g)
end

macro newtontrace(stepnorm)
    esc(quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(cache.x)
                dt["f(x)"] = copy(value(df))
                dt["g(x)"] = copy(jacobian(df))
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

function newton_{T}(df::OnceDifferentiable,
                    x0::AbstractArray{T},
                    x_tol::T,
                    f_tol::T,
                    iterations::Integer,
                    store_trace::Bool,
                    show_trace::Bool,
                    extended_trace::Bool,
                    linesearch!,
                    cache = NewtonCache(df),
                    linsolve! = (x, A, b) -> copy!(x, A\b))
    nlsolve(df, x0,
            Newton(linesearch!, linsolve!),
            Options(x_tol, f_tol, iterations, store_trace, show_trace, extended_trace),
            cache)
end
function nlsolve{T}(df, x0::AbstractArray{T}, method::Newton,
                 options::Options = Options(),
                 cache = NewtonCache(df))

    @unpack x_tol, f_tol, store_trace, show_trace, extended_trace,
            iterations = options
    x_tol, f_tol = T(x_tol), T(f_tol)

    @unpack linsolve!, linesearch! = method

    n = length(x0)
    copy!(cache.x, x0)
    value_jacobian!!(df, cache.x)
    check_isfinite(value(df))
    vecvalue = vec(value(df))
    it = 0
    x_converged, f_converged, converged = assess_convergence(value(df), f_tol)
    x_ls = copy(cache.x)
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @newtontrace convert(T, NaN)

    # Create objective function for the linesearch.
    # This function is defined as fo(x) = 0.5 * f(x) ⋅ f(x) and thus
    # has the gradient ∇fo(x) = ∇f(x) ⋅ f(x)
    function fo(xlin)
        value!(df, xlin)
        vecdot(value(df), value(df)) / 2
    end

    # The line search algorithm will want to first compute ∇fo(xₖ).
    # We have already computed ∇f(xₖ) and it is possible that it
    # is expensive to recompute.
    # We solve this using the already computed ∇f(xₖ)
    # in case of the line search asking us for the gradient at xₖ.
    function go!(storage, xlin)
        value_jacobian!(df, xlin)
        At_mul_B!(vec(storage), jacobian(df), vecvalue)
    end
    function fgo!(storage, xlin)
        value_jacobian!(df, xlin)
        linsolve!(vec(storage), jacobian(df), vecvalue)
        vecdot(value(df), value(df)) / 2
    end
    dfo = OnceDifferentiable(fo, go!, fgo!, cache.x, real(zero(T)))

    while !converged && it < iterations

        it += 1

        if it > 1
            jacobian!(df, cache.x)
        end

        try
            At_mul_B!(vec(cache.g), jacobian(df), vec(value(df)))
            linsolve!(cache.p, jacobian(df), vec(value(df)))
            scale!(cache.p, -1)
        catch e
            if isa(e, Base.LinAlg.LAPACKException) || isa(e, Base.LinAlg.SingularException)
                # Modify the search direction if the jacobian is singular
                # FIXME: better selection for lambda, see Nocedal & Wright p. 289
                fjac2 = jacobian(df)'*jacobian(df)
                lambda = convert(T,1e6)*sqrt(n*eps())*norm(fjac2, 1)
                linsolve!(cache.p, -(fjac2 + lambda*eye(n)), vec(cache.g))
            else
                throw(e)
            end
        end

        copy!(cache.xold, cache.x)

        value_gradient!(dfo, cache.x)

        alpha, ϕalpha = linesearch!(dfo, cache.x, cache.p, one(T), x_ls, value(dfo), vecdot(cache.g, cache.p))
        # fvec is here also updated in the linesearch so no need to call f again.
        copy!(cache.x, x_ls)
        x_converged, f_converged, converged = assess_convergence(cache.x, cache.xold, value(df), x_tol, f_tol)

        @newtontrace sqeuclidean(cache.x, cache.xold)
    end

    return SolverResults("Newton with line-search",
                         x0, copy(cache.x), vecnorm(value(df), Inf),
                         it, x_converged, x_tol, f_converged, f_tol, tr,
                         first(df.f_calls), first(df.df_calls))
end

function newton{T}(df::OnceDifferentiable,
                   x0::AbstractArray{T},
                   x_tol::Real,
                   f_tol::Real,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool,
                   linesearch,
                   cache = NewtonCache(df))
    newton_(df, x0, convert(T, x_tol), convert(T, f_tol), iterations, store_trace, show_trace, extended_trace, linesearch)
end
