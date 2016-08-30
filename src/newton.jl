function no_linesearch!(dfo, xold, p, x, gr, lsr, alpha, mayterminate)
    @simd for i in eachindex(x)
        @inbounds x[i] = xold[i] + p[i]
    end
    dfo.f(x)
    return 0.0, 0, 0
end

macro newtontrace(stepnorm)
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["f(x)"] = copy(fvec)
                dt["g(x)"] = copy(fjac)
            end
            update!(tr,
                    it,
                    maximum(abs(fvec)),
                    $stepnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end

function newton_{T}(df::AbstractDifferentiableMultivariateFunction,
                   initial_x::Vector{T},
                   xtol::T,
                   ftol::T,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool,
                   linesearch!::Function)

    x = copy(initial_x)
    nn = length(x)
    xold = fill(convert(T, NaN), nn)
    fvec = Array(T, nn)
    fjac = alloc_jacobian(df, T, nn)
    Tjac = typeof(fjac)
    if Tjac <: StridedVecOrMat
        lujac = alloc_jacobian(df, T, nn)
    end

    p = Array(T, nn)
    g = Array(T, nn)
    gr = Array(T, nn)

    # Count function calls
    f_calls::Int, g_calls::Int = 0, 0

    df.fg!(x, fvec, fjac)
    f_calls += 1
    g_calls += 1

    check_isfinite(fvec)

    it = 0
    x_converged, f_converged, converged = assess_convergence(x, xold, fvec, xtol, ftol)

    # FIXME: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = Optim.LineSearchResults(T)

    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @newtontrace convert(T, NaN)

    # Create objective function for the linesearch.
    # This function is defined as fo(x) = 0.5 * f(x) ⋅ f(x) and thus
    # has the gradient ∇fo(x) = ∇f(x) ⋅ f(x)
    function fo(xlin::Vector)
        if xlin != xold
            df.f!(xlin, fvec)
            f_calls += 1
        end
        return(dot(fvec, fvec) / 2)
    end

    # The line search algorithm will want to first compute ∇fo(xₖ).
    # We have already computed ∇f(xₖ) and it is possible that it
    # is expensive to recompute.
    # We solve this using the already computed ∇f(xₖ)
    # in case of the line search asking us for the gradient at xₖ.
    function go!(xlin::Vector, storage::Vector)
        if xlin == xold
            At_mul_B!(storage, fjac, fvec)
        # Else we need to recompute it.
        else
            df.fg!(xlin, fvec, fjac)
            f_calls += 1
            g_calls += 1
            At_mul_B!(storage, fjac, fvec)
        end
    end
    function fgo!(xlin::Vector, storage::Vector)
        go!(xlin, storage)
        return(dot(fvec, fvec) / 2)
    end

    dfo = DifferentiableFunction(fo, go!, fgo!)

    while !converged && it < iterations

        it += 1

        if it > 1
            df.g!(x, fjac)
            g_calls += 1
        end

        try
            if Tjac <: StridedVecOrMat
                copy!(p, fvec)
                copy!(lujac, fjac)
                A_ldiv_B!(lufact!(lujac), p)
            else
                p = fjac \ fvec
            end
            scale!(p, -1)
        catch e
            if isa(e, Base.LinAlg.LAPACKException)
                # Modify the search direction if the jacobian is singular
                # FIXME: better selection for lambda, see Nocedal & Wright p. 289
                fjac2 = Ac_mul_B(fjac, fjac)
                lambda = convert(T,1e6)*sqrt(nn*eps())*norm(fjac2, 1)
                g = Ac_mul_B(fjac, fvec)
                p = -(fjac2 + lambda*eye(nn))\g
            else
                throw(e)
            end
        end

        copy!(xold, x)

        Optim.clear!(lsr)
        push!(lsr, zero(T), dot(fvec,fvec)/2, dot(g, p))

        alpha, f_calls_update, g_calls_update =
            linesearch!(dfo, xold, p, x, gr, lsr, one(T), mayterminate)

        # fvec is here also updated in the linesearch! so no need to call f again.

        x_converged, f_converged, converged = assess_convergence(x, xold, fvec, xtol, ftol)

        @newtontrace sqeuclidean(x, xold)
    end

    return SolverResults("Newton with line-search",
                         initial_x, x, norm(fvec, Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         f_calls, g_calls)
end

function newton{T}(df::AbstractDifferentiableMultivariateFunction,
                   initial_x::Vector{T},
                   xtol::Real,
                   ftol::Real,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool,
                   linesearch!::Function)
    newton_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace, linesearch!)
end
