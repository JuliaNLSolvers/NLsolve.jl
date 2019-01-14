# Solvers for mixed complementarity problems

macro reformulate(df)
    esc(quote
        if reformulation == :smooth
            rf = mcp_smooth($df, lower, upper)
        elseif reformulation == :minmax
            rf = mcp_minmax($df, lower, upper)
        else
            throw(ArgumentError("Unknown reformulation $reformulation"))
        end
    end)
end

function mcpsolve(df::OnceDifferentiable,
                  lower::AbstractArray{T},
                  upper::AbstractArray{T},
                  initial_x::AbstractArray{T};
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = zero(T),
                  ftol::Union{Real,AbstractArray{<:Real}} = convert(T,1e-8),
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch = LineSearches.BackTracking(),
                  factor::Real = one(T),
                  autoscale::Bool = true) where T

    @reformulate df
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch = linesearch, factor = factor, autoscale = autoscale)
end

function mcpsolve(f,
                  j,
                  lower::AbstractArray{T},
                  upper::AbstractArray{T},
                  initial_x::AbstractArray{T};
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = zero(T),
                  ftol::Union{Real,AbstractArray{<:Real}} = convert(T,1e-8),
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch = LineSearches.BackTracking(),
                  factor::Real = one(T),
                  autoscale = true,
                  inplace = true) where T
    if inplace
        df = OnceDifferentiable(f, initial_x, initial_x)
    else
        df = OnceDifferentiable(not_in_place(f, j)..., initial_x, initial_x)
    end
    @reformulate df
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch = linesearch, factor = factor, autoscale = autoscale)
end

function mcpsolve(f,
                  lower::AbstractArray{T},
                  upper::AbstractArray{T},
                  initial_x::AbstractArray{T};
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = zero(T),
                  ftol::Union{Real,AbstractArray{<:Real}} = convert(T,1e-8),
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch = LineSearches.BackTracking(),
                  factor::Real = one(T),
                  autoscale::Bool = true,
                  autodiff = :central,
                  inplace = true) where T
    if inplace
        df = OnceDifferentiable(f, initial_x, initial_x, autodiff)
    else
        df = OnceDifferentiable(not_in_place(f), initial_x, initial_x, autodiff)
    end

    @reformulate df
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch = linesearch, factor = factor, autoscale = autoscale)
end
