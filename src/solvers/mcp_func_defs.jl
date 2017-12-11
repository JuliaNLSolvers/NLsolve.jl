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

function mcpsolve(df::TDF,
                  lower::Vector,
                  upper::Vector,
                  initial_x::AbstractArray{T};
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = zero(T),
                  ftol::Real = convert(T,1e-8),
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch! = LineSearches.BackTracking(),
                  factor::Real = one(T),
                  autoscale::Bool = true) where {TDF <: OnceDifferentiable, T}

    @reformulate df
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end

function mcpsolve{T}(f!,
                  j!,
                  lower::Vector,
                  upper::Vector,
                  initial_x::AbstractArray{T};
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = zero(T),
                  ftol::Real = convert(T,1e-8),
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch! = LineSearches.BackTracking(),
                  factor::Real = one(T),
                  autoscale::Bool = true)
    @reformulate OnceDifferentiable(f!, j!, similar(initial_x), initial_x)
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end

function mcpsolve{T}(f!,
                  lower::Vector,
                  upper::Vector,
                  initial_x::AbstractArray{T};
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = zero(T),
                  ftol::Real = convert(T,1e-8),
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch! = LineSearches.BackTracking(),
                  factor::Real = one(T),
                  autoscale::Bool = true,
                  autodiff::Bool = false)
    if !autodiff
        df = OnceDifferentiable(f!, similar(initial_x), initial_x)
    else
        df = OnceDifferentiable(f!, initial_x, initial_x)
    end
    @reformulate df
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end
