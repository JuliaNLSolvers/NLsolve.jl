struct Newton{LS} <: AbstractNLSolver
    linesearch!::LS
end
Newton(;linesearch = LineSearches.BackTracking()) = Newton(linesearch)
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
                dt["x"] = copy(x)
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
                    initial_x::AbstractArray{T},
                    xtol::T,
                    ftol::T,
                    iterations::Integer,
                    store_trace::Bool,
                    show_trace::Bool,
                    extended_trace::Bool,
                    linesearch)
    # setup
    x = vec(copy(initial_x))
    n = length(x)
    xold = similar(x)
    p = Array{T}(n)
    g = Array{T}(n)
    value_jacobian!!(df, x)

    check_isfinite(value(df))

    it = 0
    x_converged, f_converged, converged = assess_convergence(value(df), ftol)

    # FIXME: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearches.LineSearchResults(T)

    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @newtontrace convert(T, NaN)

    # Create objective function for the linesearch.
    # This function is defined as fo(x) = 0.5 * f(x) ⋅ f(x) and thus
    # has the gradient ∇fo(x) = ∇f(x) ⋅ f(x)
    function fo(xlin::AbstractVector)
        if xlin != xold
            value!(df, xlin)
        end
        vecdot(value(df), value(df)) / 2
    end

    # The line search algorithm will want to first compute ∇fo(xₖ).
    # We have already computed ∇f(xₖ) and it is possible that it
    # is expensive to recompute.
    # We solve this using the already computed ∇f(xₖ)
    # in case of the line search asking us for the gradient at xₖ.
    function go!(storage::AbstractVector, xlin::AbstractVector)
        if xlin != xold
            value_jacobian!(df, xlin)
        end
        At_mul_B!(storage, jacobian(df), vec(value(df)))
    end
    function fgo!(storage::AbstractVector, xlin::AbstractVector)
        go!(storage, xlin)
        vecdot(value(df), value(df)) / 2
    end
    dfo = OnceDifferentiable(fo, go!, fgo!, x, real(zero(T)))

    while !converged && it < iterations

        it += 1

        if it > 1
            jacobian!(df, x)
        end

        try
            At_mul_B!(g, jacobian(df), vec(value(df)))
            p = jacobian(df)\vec(value(df))
            scale!(p, -1)
        catch e
            if isa(e, Base.LinAlg.LAPACKException) || isa(e, Base.LinAlg.SingularException)
                # Modify the search direction if the jacobian is singular
                # FIXME: better selection for lambda, see Nocedal & Wright p. 289
                fjac2 = jacobian(df)'*jacobian(df)
                lambda = convert(T,1e6)*sqrt(n*eps())*norm(fjac2, 1)
                p = -(fjac2 + lambda*eye(n))\vec(g)
            else
                throw(e)
            end
        end

        copy!(xold, x)

        LineSearches.clear!(lsr)
        push!(lsr, zero(T), vecdot(value(df),value(df))/2, vecdot(g, p))

        alpha = linesearch(dfo, xold, p, x, lsr, one(T), mayterminate)

        # fvec is here also updated in the linesearch so no need to call f again.

        x_converged, f_converged, converged = assess_convergence(x, xold, value(df), xtol, ftol)

        @newtontrace sqeuclidean(x, xold)
    end

    return SolverResults("Newton with line-search",
                         initial_x, reshape(x, size(initial_x)...), vecnorm(value(df), Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         first(df.f_calls), first(df.df_calls))
end

function newton{T}(df::OnceDifferentiable,
                   initial_x::AbstractArray{T},
                   xtol::Real,
                   ftol::Real,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool,
                   linesearch)
    newton_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace, linesearch)
end
