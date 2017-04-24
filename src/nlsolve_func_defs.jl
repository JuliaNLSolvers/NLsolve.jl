function nlsolve{T}(df::AbstractDifferentiableMultivariateFunction,
                 initial_x::Vector{T};
                 method::Symbol = :trust_region,
                 xtol::Real = zero(T),
                 ftol::Real = convert(T,1e-8),
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch!::Function = no_linesearch!,
                 factor::Real = one(T),
                 autoscale::Bool = true,
                 kwargs...)
    if extended_trace
        show_trace = true
    end
    if show_trace
        @printf "Iter     f(x) inf-norm    Step 2-norm \n"
        @printf "------   --------------   --------------\n"
    end
    if method == :newton
        newton(df, initial_x, xtol, ftol, iterations,
               store_trace, show_trace, extended_trace, linesearch!)
    elseif method == :trust_region
        trust_region(df, initial_x, xtol, ftol, iterations,
                     store_trace, show_trace, extended_trace, factor,
                     autoscale)

    elseif method == :df_sane
        df_sane(
            df.f!, initial_x; ftol=ftol, xtol=xtol, iterations=iterations,
            store_trace=store_trace, show_trace=show_trace,
            extended_trace=extended_trace,
            kwargs...
        )
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function nlsolve{T}(f!::Function,
                    g!::Function,
                    initial_x::Vector{T};
                    kwargs...)
    nlsolve(DifferentiableMultivariateFunction(f!, g!), initial_x; kwargs...)
end

function nlsolve{T}(f!::Function,
                 initial_x::Vector{T};
                 autodiff::Bool=false,
                 kwargs...)
    if !autodiff
        df = DifferentiableMultivariateFunction(f!)
    else
        df = NLsolve.autodiff(f!, initial_x)
    end
    nlsolve(df, initial_x; kwargs...)
end
