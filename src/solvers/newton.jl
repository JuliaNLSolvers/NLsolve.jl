struct Newton
end
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
function no_linesearch(dfo, xold, p, x, lsr, alpha, mayterminate)
    @simd for i in eachindex(x)
        @inbounds x[i] = xold[i] + p[i]
    end
    dfo.f(x)
    return 0.0, 0, 0
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

function newton_(df::OnceDifferentiable,
                    initial_x::AbstractArray{T},
                    xtol::T,
                    ftol::T,
                    iterations::Integer,
                    store_trace::Bool,
                    show_trace::Bool,
                    extended_trace::Bool,
                    linesearch,
                    cache = NewtonCache(df)) where T
    n = length(initial_x)
    copyto!(cache.x, initial_x)
    value_jacobian!!(df, cache.x)
    check_isfinite(value(df))
    vecvalue = vec(value(df))
    it = 0
    x_converged, f_converged, converged = assess_convergence(value(df), ftol)
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
        mul!(vec(storage), transpose(jacobian(df)), vecvalue)
    end
    function fgo!(storage, xlin)
        value_jacobian!(df, xlin)
        mul!(vec(storage), transpose(jacobian(df)), vecvalue)
        vecdot(value(df), value(df)) / 2
    end
    dfo = OnceDifferentiable(fo, go!, fgo!, cache.x, real(zero(T)))

    while !converged && it < iterations

        it += 1

        if it > 1
            jacobian!(df, cache.x)
        end

        try
            mul!(vec(cache.g), transpose(jacobian(df)), vec(value(df)))
            copyto!(cache.p, jacobian(df)\vec(value(df)))
            rmul!(cache.p, -1)
        catch e
            if isa(e, LAPACKException) || isa(e, SingularException)
                # Modify the search direction if the jacobian is singular
                # FIXME: better selection for lambda, see Nocedal & Wright p. 289
                fjac2 = jacobian(df)'*jacobian(df)
                lambda = convert(T,1e6)*sqrt(n*eps())*norm(fjac2, 1)
                cache.p .= -(fjac2 + lambda*eye(n))\vec(cache.g)
            else
                throw(e)
            end
        end

        copyto!(cache.xold, cache.x)

        value_gradient!(dfo, cache.x)

        alpha, ϕalpha = linesearch(dfo, cache.x, cache.p, one(T), x_ls, value(dfo), vecdot(cache.g, cache.p))
        # fvec is here also updated in the linesearch so no need to call f again.
        copyto!(cache.x, x_ls)
        x_converged, f_converged, converged = assess_convergence(cache.x, cache.xold, value(df), xtol, ftol)

        @newtontrace sqeuclidean(cache.x, cache.xold)
    end

    return SolverResults("Newton with line-search",
                         initial_x, copy(cache.x), norm(value(df), Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         first(df.f_calls), first(df.df_calls))
end

function newton(df::OnceDifferentiable,
                   initial_x::AbstractArray{T},
                   xtol::Real,
                   ftol::Real,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool,
                   linesearch,
                   cache = NewtonCache(df)) where T
    newton_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace, linesearch)
end
