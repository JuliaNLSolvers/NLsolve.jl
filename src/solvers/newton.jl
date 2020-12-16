struct Newton
end
struct NewtonCache{Tx} <: AbstractSolverCache
    x::Tx
    xold::Tx
    p::Tx
    g::Tx
end
function NewtonCache(df)
    x = copy(df.x_f)
    xold = copy(x)
    p = copy(x)
    g = copy(x)
    NewtonCache(x, xold, p, g)
end

function newtontrace(stepnorm, tracing, extended_trace, cache, df, it, tr, store_trace, show_trace)
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
                stepnorm,
                dt,
                store_trace,
                show_trace)
    end
end

function newton_(df::OnceDifferentiable,
                 initial_x::AbstractArray{T},
                 xtol::Real,
                 ftol::Real,
                 iterations::Integer,
                 store_trace::Bool,
                 show_trace::Bool,
                 extended_trace::Bool,
                 linesearch,
                 linsolve,
                 cache = NewtonCache(df)) where T
    n = length(initial_x)
    copyto!(cache.x, initial_x)
    value_jacobian!!(df, cache.x)
    check_isfinite(value(df))
    vecvalue = vec(value(df))
    it = 0
    x_converged, f_converged = assess_convergence(initial_x, cache.xold, value(df), NaN, ftol)
    stopped = any(isnan, cache.x) || any(isnan, value(df)) ? true : false

    converged = x_converged || f_converged
    x_ls = copy(cache.x)
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    newtontrace(convert(real(T), NaN), tracing, extended_trace, cache, df, it, tr, store_trace, show_trace)

    # Create objective function for the linesearch.
    # This function is defined as fo(x) = 0.5 * f(x) ⋅ f(x) and thus
    # has the gradient ∇fo(x) = ∇f(x) ⋅ f(x)
    function fo(xlin)
        value!(df, xlin)
        dot(value(df), value(df)) / 2
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
        dot(value(df), value(df)) / 2
    end
    dfo = OnceDifferentiable(fo, go!, fgo!, cache.x, zero(real(T)))

    while !stopped && !converged && it < iterations

        it += 1

        if it > 1
            value_jacobian!(df, cache.x)
        end

        try
            linsolve(cache.p, jacobian(df), vec(value(df)))
            rmul!(cache.p, -1)
        catch e
            if isa(e, LAPACKException) || isa(e, SingularException)
                # Modify the search direction if the jacobian is singular
                # FIXME: better selection for lambda, see Nocedal & Wright p. 289
                fjac2 = jacobian(df)'*jacobian(df)
                lambda = convert(real(T),1e6)*sqrt(n*eps())*norm(fjac2, 1)
                linsolve(cache.p, -(fjac2 + lambda * I), vec(value(df)))
            else
                throw(e)
            end
        end

        copyto!(cache.xold, cache.x)


        
        if linesearch isa Static
            x_ls .= cache.x .+ cache.p
            value_jacobian!(df, x_ls)
            alpha, ϕalpha = one(real(T)), value(dfo)
        else
            mul!(vec(cache.g), transpose(jacobian(df)), vec(value(df)))
            value_gradient!(dfo, cache.x)
            alpha, ϕalpha = linesearch(dfo, cache.x, cache.p, one(real(T)), x_ls, value(dfo), dot(cache.g, cache.p))
        end
        # fvec is here also updated in the linesearch so no need to call f again.
        copyto!(cache.x, x_ls)
        x_converged, f_converged = assess_convergence(cache.x, cache.xold, value(df), xtol, ftol)
        stopped = any(isnan, cache.x) || any(isnan, value(df)) ? true : false

        converged = x_converged || f_converged
        newtontrace(sqeuclidean(cache.x, cache.xold), tracing, extended_trace, cache, df, it, tr, store_trace, show_trace)
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
                cache = NewtonCache(df);
                linsolve=(x, A, b) -> copyto!(x, A\b)) where T
    newton_(df, initial_x, convert(real(T), xtol), convert(real(T), ftol), iterations, store_trace, show_trace, extended_trace, linesearch, linsolve, cache)
end
