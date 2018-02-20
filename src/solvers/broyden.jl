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

function broyden_{T}(df::OnceDifferentiable,
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
    value!(df, x)
    n = length(x)
    xold = similar(x)
    fold = similar(value(df))
    p = Array{T}(n)
    g = Array{T}(n)
    Jinv = eye(T, n, n)
    check_isfinite(value(df))
    vecvalue = vec(value(df))
    it = 0
    x_converged, f_converged, converged = assess_convergence(value(df), ftol)

    # FIXME: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
  #  lsr = LineSearches.LineSearchResults(T)

    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @broydentrace convert(T, NaN)

    #Create objective function for the linesearch.
    # This function is defined as fo(x) = 0.5 * f(x) ⋅ f(x) and thus
    # has the gradient ∇fo(x) = ∇f(x) ⋅ f(x)
   # function fo(xlin::AbstractVector)
    #    if xlin != xold
     #       value!(df, xlin)
      #  end
       # vecdot(value(df), value(df)) / 2
   # end

    # The line search algorithm will want to first compute ∇fo(xₖ).
    # We have already computed ∇f(xₖ) and it is possible that it
    # is expensive to recompute.
    # We solve this using the already computed ∇f(xₖ)
    # in case of the line search asking us for the gradient at xₖ.
#    function go!(storage::AbstractVector, xlin::AbstractVector)
 #       if xlin != xold
  #          value_jacobian!(df, xlin)
   #     end
    #    At_mul_B!(storage, jacobian(df), vec(value(df)))
    #end
 #   function fgo!(storage::AbstractVector, xlin::AbstractVector)
  #      go!(storage, xlin)
   #     vecdot(value(df), value(df)) / 2
   # end
 #   dfo = OnceDifferentiable(fo, go!, fgo!, x, real(zero(T)))

    while !converged && it < iterations

        it += 1

        if it > 1
            Δx = x-xold
            Δf = value(df)-fold
            Jinv = Jinv + ((Δx - Jinv *Δf)/(Δx'Jinv*Δf))*Δx'Jinv
        end

        copy!(xold, x)
        fold = copy(value(df))
        p = - Jinv*fold

        function lsobj(α)
            value!(df, x+α*p)
            vecdot(value(df), value(df))/2
        end
        function lsgs(α)
            value_jacobian!(df, x+α*p)
            vecdot(At_mul_B(jacobian(df), vecvalue), p)
        end

        α = T(1.0)
        lsa = lsobj(α)
        ls1 = lsobj(1.0)
        lsg1 = lsobj(1.0)
        while lsobj(α) < ls1 + α* lsg1
            α = α/2
            lsa = lsobj(α)
        end
        println(α)
        x = xold + α*p
        value!(df, x)

        x_converged, f_converged, converged = assess_convergence(x, xold, value(df), xtol, ftol)

        @broydentrace sqeuclidean(x, xold)
    end
@show initial_x
@show reshape(x, size(initial_x)...)
    return SolverResults("broyden without line-search",
                         initial_x, reshape(x, size(initial_x)...), vecnorm(value(df), Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         first(df.f_calls), first(df.df_calls))
end

function broyden{T}(df::OnceDifferentiable,
                   initial_x::AbstractArray{T},
                   xtol::Real,
                   ftol::Real,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool,
                   linesearch)
    broyden_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace, linesearch)
end
