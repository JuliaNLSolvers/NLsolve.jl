function fixedpoint(df::TDF,
                 initial_x::AbstractArray{T};
                 method::Symbol = :simple_iteration,
                 xtol::Real = zero(T),
                 ftol::Real = convert(T,1e-8),
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch = LineSearches.Static(),
                 factor::Real = one(T),
                 autoscale::Bool = true,
                 m::Integer = 0,
                 beta::Real = 1.0) where {T, TDF <: OnceDifferentiable}
    if extended_trace
        show_trace = true
    end
    if show_trace
        @printf "Iter     f(x) inf-norm    Step 2-norm \n"
        @printf "------   --------------   --------------\n"
    end
    if method == :simple_iteration
        simple_iteration(df, initial_x, xtol, ftol, iterations)
#    elseif method == :newton
#newton(df, initial_x, xtol, ftol, iterations,
#               store_trace, show_trace, extended_trace, linesearch)
#    elseif method == :trust_region
#        trust_region(df, initial_x, xtol, ftol, iterations,
#                     store_trace, show_trace, extended_trace, factor,
#                     autoscale)
#    elseif method == :anderson
#        anderson(df, initial_x, xtol, ftol, iterations,
#                 store_trace, show_trace, extended_trace, m, beta)
    else
        throw(ArgumentError("Method $method unknown or not implemented"))
    end
end

function fixedpoint{T}(f,
                 initial_x::AbstractArray{T};
                 method::Symbol = :simple_iteration,
                 xtol::Real = zero(T),
                 ftol::Real = convert(T,1e-8),
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch = LineSearches.Static(),
                 factor::Real = one(T),
                 autoscale::Bool = true,
                 m::Integer = 0,
                 beta::Real = 1.0,
                 autodiff = :central,
                 inplace = true)
    if inplace
        df = OnceDifferentiable(f, initial_x, initial_x, autodiff)
    else
        df = OnceDifferentiable(not_in_place(f), initial_x, initial_x, autodiff)
    end

    fixedpoint(df,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch = linesearch, factor = factor, autoscale = autoscale,
            m = m, beta = beta)
end


function fixedpoint(f!,
                j!,
                initial_x::AbstractArray{T};
                method::Symbol = :simple_iteration,
                xtol::Real = zero(T),
                ftol::Real = convert(T, 1e-8),
                iterations::Integer = 1_000,
                store_trace::Bool = false,
                show_trace::Bool = false,
                extended_trace::Bool = false,
                linesearch = LineSearches.Static(),
                factor::Real = one(T),
                autoscale::Bool = true,
                m::Integer = 0,
                beta::Real = 1.0,
                inplace = true) where T
    if inplace
        df = OnceDifferentiable(f!, j!, initial_x, initial_x)
    else
        df = OnceDifferentiable(not_in_place(f!, j!)..., initial_x, initial_x)
    end
    fixedpoint(df,
    initial_x, method = method, xtol = xtol, ftol = ftol,
    iterations = iterations, store_trace = store_trace,
    show_trace = show_trace, extended_trace = extended_trace,
    linesearch = linesearch, factor = factor, autoscale = autoscale,
    m = m, beta = beta)
end

function fixedpoint(f!,
                j!,
                fj!,
                initial_x::AbstractArray{T};
                method::Symbol = :simple_iteration,
                xtol::Real = zero(T),
                ftol::Real = convert(T, 1e-8),
                iterations::Integer = 1_000,
                store_trace::Bool = false,
                show_trace::Bool = false,
                extended_trace::Bool = false,
                linesearch = LineSearches.Static(),
                factor::Real = one(T),
                autoscale::Bool = true,
                m::Integer = 0,
                beta::Real = 1.0,
                inplace = true) where T
    if inplace
        df = OnceDifferentiable(f!, j!, fj!, initial_x, initial_x)
    else
        df = OnceDifferentiable(not_in_place(f!, j!, fj!)..., initial_x, initial_x)
    end
    fixedpoint(df,
    initial_x, method = method, xtol = xtol, ftol = ftol,
    iterations = iterations, store_trace = store_trace,
    show_trace = show_trace, extended_trace = extended_trace,
    linesearch = linesearch, factor = factor, autoscale = autoscale,
    m = m, beta = beta)
end
