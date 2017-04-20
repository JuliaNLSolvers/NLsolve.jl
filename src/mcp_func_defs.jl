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

function mcpsolve{T}(df::AbstractDifferentiableMultivariateFunction,
                  lower::Vector,
                  upper::Vector,
                  initial_x::Vector{T};
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = zero(T),
                  ftol::Real = convert(T,1e-8),
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = LineSearches.backtracking!,
                  factor::Real = one(T),
                  autoscale::Bool = true,
                  kwargs...)
    if method == :lm
        verbosity = 0
        if show_trace
            verbosity += 1
        end
        if extended_trace
            verbosity += 1
        end
        return lmmcp(
            df, lower, upper, initial_x;
            ftol=ftol, iterations=iterations, verbosity=verbosity,
            kwargs...
         )
    end

    @reformulate df
    nlsolve(rf,
            initial_x; method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end

function mcpsolve{T}(f!::Function,
                  g!::Function,
                  lower::Vector,
                  upper::Vector,
                  initial_x::Vector{T};
                  kwargs...)
    df = DifferentiableMultivariateFunction(f!, g!)
    mcpsolve(df, lower, upper, initial_x; kwargs...)
end

function mcpsolve{T}(f!::Function,
                  lower::Vector,
                  upper::Vector,
                  initial_x::Vector{T};
                  autodiff::Bool=false,
                  kwargs...)
    if !autodiff
        df = DifferentiableMultivariateFunction(f!)
    else
        df = NLsolve.autodiff(f!, initial_x)
    end
    mcpsolve(df, lower, upper, initial_x; kwargs...)
end
