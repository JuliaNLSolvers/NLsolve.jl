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

function broyden_(df::Union{NonDifferentiable, OnceDifferentiable},
                    initial_x::AbstractArray{T},
                    xtol::T,
                    ftol::Union{T,AbstractArray{T}},
                    iterations::Integer,
                    store_trace::Bool,
                    show_trace::Bool,
                    extended_trace::Bool,
                    linesearch) where T
    # setup
    n = length(initial_x)
    x = vec(copy(initial_x))

    value!!(df, x)

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
    max_resets = 3
    resets = 0
    while !converged && it < iterations

        it += 1

        copyto!(xold, x)
        copyto!(fold, value(df))
        p = - Jinv*fold


        ρ = T(0.9)
        σ₂ = T(0.001)
        x = xold + p
        value!(df, x)
        if norm(value(df), 2) <= ρ*norm(fold, 2) - σ₂*norm(p, 2)^2 # condition 2.7
            α = T(1.0)
        else
            α = approximate_norm_descent(x->value!(df, x), x, p)
            x = xold + α*p
        end

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
                         first(df.f_calls), 0)
end

function broyden(df::Union{NonDifferentiable, OnceDifferentiable},
                   initial_x::AbstractArray{T},
                   xtol::Real,
                   ftol::Union{Real,AbstractArray{<:Real}},
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool,
                   linesearch) where T
    broyden_(df, initial_x, convert(T, xtol), convert.(T, ftol), iterations, store_trace, show_trace, extended_trace, linesearch)
end

# A derivative-free line search and global convergence
# of Broyden-like method for nonlinear equations
# by Dong-Hui Li & Masao Fukushima
# https://doi.org/10.1080/10556780008805782
function approximate_norm_descent(F, x::AbstractArray{T}, p;
                                  lambda_0=T(1.0), beta=T(0.5),
                                  sigma_1=T(0.001), eta=T(0.1),
                                  nan_max_iter=5, max_iter=50) where T
    β, σ₁, η = beta, sigma_1, eta
    λ₂ = lambda_0
    λ₁ = λ₂

    # Calculate norms
    Fx_norm = norm(F(x), 2)
    if isnan(Fx_norm)
        throw(ErrorException("Exiting line search: 2-norm of current residuals is not a number. Value is: ", Fx_norm))
    elseif !isfinite(Fx_norm)
        throw(ErrorException("Exiting line search: 2-norm of current residuals is not finite. Value is: ", Fx_norm))
    end

    Fxλp_norm = norm(F(x - λ₂*p), 2)

    # If the step produces something that is not finite or a number, backtrack
    # a maximum number of nan_max_iter number of times to try to find a suitable
    # range of values for which F(x + λp) is well-behaved
    if isnan(Fxλp_norm) || !isfinite(Fxλp_norm)
         λ₁, λ₂ = nantrack(F, x, λ₂, p; β=β, nan_max_iter=nan_max_iter)
    end

    j = 0
    converged = norm(F(x + λ₂*p), 2) <= Fx_norm - σ₁*norm(λ₂*p, 2)^2 + η*norm(F(x), 2)

    while j < max_iter && !converged
        j += 1

        λ₁, λ₂ = λ₂, β*λ₂
        Fxλp_norm = norm(F(x - λ₂*p), 2)

        converged = norm(F(x + λ₂*p), 2) <= Fx_norm - σ₁*norm(λ₂*p, 2)^2 + η*norm(F(x), 2)
    end
    if j >= max_iter && !converged
        throw(ErrorException("Exiting line search: failed to satisfy condition (2.4)."))
    end
    λ₂
end


function nantrack(F, x, λ₂, p; β=0.5, nan_max_iter=5)
    i = 0
    nan_converged = false
    while i  <= nan_max_iter && !nan_converged
        i +=1

        λ₁, λ₂ = λ₂, β*λ₂
        Fxλp_norm = norm(F(x - λ₂*p), 2)

        converged = !isnan(Fxλp_norm)
    end
    if i >= nan_max_iter
        throw(ErrorException("Exiting line search: failed to find finite 2-norm of residuals in $nan_max_iter trials."))
    end
    return λ₁, λ₂
end
