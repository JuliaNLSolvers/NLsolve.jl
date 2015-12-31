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

function create_objective_function(df::AbstractDifferentiableMultivariateFunction,
                                   T::Type, nn::Integer)
    fvec2 = Array(T, nn)
    fjac2 = alloc_jacobian(df, T, nn)
    function fo(x::Vector{T})
        df.f!(x, fvec2)
        return(dot(fvec2, fvec2)/2)
    end
    function go!(x::Vector{T}, storage::Vector{T})
        df.fg!(x, fvec2, fjac2)
        copy!(storage, fjac2'*fvec2)
    end
    function fgo!(x::Vector{T}, storage::Vector{T})
        df.fg!(x, fvec2, fjac2)
        copy!(storage, fjac2'*fvec2)
        return(dot(fvec2, fvec2)/2)
    end

    return(DifferentiableFunction(fo, go!, fgo!))
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
    p = Array(T, nn)
    g = Array(T, nn)
    gr = Array(T, nn)

    # Count function calls
    f_calls, g_calls = 0, 0

    df.f!(x, fvec)
    f_calls += 1

    i = find(!isfinite(fvec))

    if !isempty(i)
        error("During the resolution of the non-linear system, the evaluation of the following equation(s) resulted in a non-finite number: $(i)")
    end

    it = 0
    x_converged, f_converged, converged = assess_convergence(x, xold, fvec, xtol, ftol)

    # FIXME: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = Optim.LineSearchResults(T)

    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @newtontrace convert(T,NaN)

    dfo = create_objective_function(df, T, nn)

    while !converged && it < iterations

        it += 1

        df.g!(x, fjac)
        g_calls += 1

        g = fjac'*fvec

        try
            p = -fjac\fvec
        catch e
            if isa(e, Base.LinAlg.LAPACKException)
                # Modify the search direction if the jacobian is singular
                # FIXME: better selection for lambda, see Nocedal & Wright p. 289
                fjac2 = fjac'*fjac
                lambda = convert(T,1e6)*sqrt(nn*eps())*norm(fjac2, 1)
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

        f_calls += f_calls_update
        g_calls += g_calls_update

        df.f!(x, fvec)
        f_calls += 1

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
