function nlsolve(df::TDF,
                 initial_x::AbstractArray{T},
                 method::AbstractSolver;
                 x_abstol::Real = zero(T),
                 f_abstol::Real = convert(T,1e-8),
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 autoscale::Bool = true) where {T, TDF <: OnceDifferentiable}

    options = Options(x_abstol = x_abstol, f_abstol = f_abstol,
                      iterations = iterations,
                      store_trace = store_trace, show_trace = show_trace,
                      extended_trace = extended_trace,
                      autoscale = autoscale)

    nlsolve(df, initial_x, method, options)
end

function nlsolve{T}(f,
                 initial_x::AbstractArray{T},
                 method::::AbstractSolver;
                 x_abstol::Real = zero(T),
                 f_abstol::Real = convert(T,1e-8),
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 autoscale::Bool = true,
                 autodiff = :central,
                 inplace = true)

    if typeof(f) <: Union{InplaceObjective, NotInplaceObjective}
        df = OnceDifferentiable(f, initial_x, initial_x)
    else
        if inplace
            df = OnceDifferentiable(f, initial_x, initial_x, autodiff)
        else
            df = OnceDifferentiable(not_in_place(f), initial_x, initial_x, autodiff)
        end
    end

    options = Options(x_abstol = x_abstol, f_abstol = f_abstol,
                      iterations = iterations,
                      store_trace = store_trace, show_trace = show_trace,
                      extended_trace = extended_trace,
                      autoscale = autoscale)

    nlsolve(df, initial_x, method, options)
end


function nlsolve(f!,
                j!,
                initial_x::AbstractArray{T},
                method::AbstractSolver;
                x_abstol::Real = zero(T),
                f_abstol::Real = convert(T,1e-8),
                iterations::Integer = 1_000,
                store_trace::Bool = false,
                show_trace::Bool = false,
                extended_trace::Bool = false,
                autoscale::Bool = true,
                autodiff = :central,
                inplace = true,
                inplace = true) where T
    if inplace
        df = OnceDifferentiable(f!, j!, initial_x, initial_x)
    else
        df = OnceDifferentiable(not_in_place(f!, j!)..., initial_x, initial_x)
    end

    options = Options(x_abstol = x_abstol, f_abstol = f_abstol,
                      iterations = iterations,
                      store_trace = store_trace, show_trace = show_trace,
                      extended_trace = extended_trace,
                      autoscale = autoscale)

    nlsolve(df, initial_x,method, options)
end

function nlsolve(f!,
                j!,
                fj!,
                x_abstol::Real = zero(T),
                f_abstol::Real = convert(T,1e-8),
                iterations::Integer = 1_000,
                store_trace::Bool = false,
                show_trace::Bool = false,
                extended_trace::Bool = false,
                autoscale::Bool = true,
                autodiff = :central,
                inplace = true) where T
    if inplace
        df = OnceDifferentiable(f!, j!, fj!, initial_x, initial_x)
    else
        df = OnceDifferentiable(not_in_place(f!, j!, fj!)..., initial_x, initial_x)
    end

    options = Options(x_abstol = x_abstol, f_abstol = f_abstol,
                      iterations = iterations,
                      store_trace = store_trace, show_trace = show_trace,
                      extended_trace = extended_trace,
                      autoscale = autoscale)

    nlsolve(df, initial_x,method, options)
end
