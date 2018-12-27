struct Broyden
end

macro broydentrace(stepnorm)
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

function broyden_(df::OnceDifferentiable,
                    initial_x::AbstractArray{T},
                    xtol::T,
                    ftol::T,
                    iterations::Integer,
                    store_trace::Bool,
                    show_trace::Bool,
                    extended_trace::Bool,
                    linesearch) where T
    # setup
    n = length(initial_x)
    x = vec(copy(initial_x))

    value!(df, x)

    vecvalue = vec(value(df))
    fold, xold = similar(vecvalue), similar(x)
    fold .= T(0)
    xold .= T(0)

    p = similar(x)
    g = similar(x)
    Jinv = Matrix{T}(I, n, n)
    check_isfinite(value(df))
    it = 0
    x_converged, f_converged, converged = assess_convergence(value(df), ftol)

    # FIXME: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
  #  lsr = LineSearches.LineSearchResults(T)

    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @broydentrace T(NaN)

    maybe_stuck = false
    max_resets = 10
    resets = 0
    while !converged && it < iterations

        it += 1

        copyto!(xold, x)
        copyto!(fold, value(df))
        p = - Jinv*fold
        function lsf(y)
            value!(df, y)
            dot(value(df), value(df))/2
        end

        function lsgs(α)
            value_jacobian!(df, x+α*p)
            inv(Jinv)'*vecvalue
        end

        α = backtracking(lsf, x, p, lsgs(T(1.0)))

        x = xold + α*p

        value!(df, x)

        Δx = x-xold
        Δf = vecvalue .- fold

        maybe_stuck = all(abs.(Δx) .<= 1e-12) || all(abs.(Δf) .<= 1e-12)

        if maybe_stuck
            Jinv = Matrix{T}(I, n, n)
            resets += 1
            if resets > max_resets
                maybe_stuck = false # to allow convergence check
            end
        else
            Jinv = Jinv + ((Δx - Jinv *Δf)/(Δx'Jinv*Δf))*Δx'Jinv
        end

        if !maybe_stuck
            x_converged, f_converged, converged = assess_convergence(x, xold, value(df), xtol, ftol)
        end

        maybe_stuck = false
        @broydentrace sqeuclidean(x, xold)
    end
    return SolverResults("broyden without line-search",
                         initial_x, copyto!(similar(initial_x), x), norm(value(df), Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         first(df.f_calls), first(df.df_calls))
end

function broyden(df::OnceDifferentiable,
                   initial_x::AbstractArray{T},
                   xtol::Real,
                   ftol::Real,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool,
                   linesearch) where T
    broyden_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace, linesearch)
end

function backtracking(f, x::AbstractArray{T}, d, ∇f_x; α_0=T(1.0),
                      ratio=T(0.5), c=T(0.001), max_iter=1000,
                      verbose=false) where T
    if verbose
        println("Entering line search with step size: ", α_0)
        println("Initial value: ", f(x))
    end

    t = -dot(d, ∇f_x)*c
    α, β = α_0, α_0

    iter = 0

    # Preface with check for NaN. As long as NaNs remain, keep backtracking
    # without checking for sufficient descent.
    f_α = f(x + α*d) # initial function value
    while !isfinite(f_α) && iter <= max_iter
        iter += 1
        β, α = α, α*ratio # backtrack according to specified ratio
        f_α = f(x + α*d) # update function value
    end

    while f(x + α*d) - f(x) >= α*t && iter <= max_iter
        iter += 1
        β, α = α, ratio*β # backtrack according to specified ratio
        if verbose
            println("α: ", α)
            println("α*t: ", α*t)
            println("Value at α: ", f(x + α*d))
        end
    end

    if iter >= max_iter
        if verbose
            println("max_iter exceeded in backtracking")
        end
    end

    if verbose
        println("Exiting line search with step size: ", α)
    end
    return α
end
