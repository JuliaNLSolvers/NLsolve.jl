
function nlsolve(df::TDF,
                 initial_x::AbstractArray{T},
                 method::TM = NewtonTrustRegion(),
                 options = Options(T)) where {T, TDF <: AbstractObjective, TM <: AbstractNLSolver}
    if options.show_trace
        @printf "Iter     f(x) inf-norm    Step 2-norm \n"
        @printf "------   --------------   --------------\n"
    end
    if TM <: Newton
        newton(df, initial_x, options.xtol, options.ftol, options.iterations,
               options.store_trace, options.show_trace, options.extended_trace, method.linesearch!)
    elseif TM <: NewtonTrustRegion
        trust_region(df, initial_x, options.xtol, options.ftol, options.iterations,
                     options.store_trace, options.show_trace, options.extended_trace, method.factor,
                     method.autoscale)
    elseif TM <: Anderson
        anderson(df, initial_x, options.xtol, options.ftol, options.iterations,
        options.store_trace, options.show_trace, options.extended_trace, method.m, method.β)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function nlsolve(f,
                 j,
                 initial_x::AbstractArray{T},
                 method::TM = NewtonTrustRegion(),
                 options = Options(T); inplace = true) where {T, TM<:AbstractNLSolver}

    if inplace
        df = OnceDifferentiable(f, j, initial_x, initial_x)
    else
        df = OnceDifferentiable(not_in_place(f, j)..., initial_x, initial_x)
    end
    nlsolve(df, initial_x, method, options)
end

function nlsolve(f,
                 initial_x::AbstractArray{T},
                 method::TM = NewtonTrustRegion(),
                 options = Options(T); inplace = true, autodiff = :forward) where {T, TM<:AbstractNLSolver}
    if inplace
        df = OnceDifferentiable(f, initial_x, initial_x, autodiff)
    else
        df = OnceDifferentiable(not_in_place(f), initial_x, initial_x, autodiff)
    end
    nlsolve(df, initial_x, method, options)
end

function nlsolve(f, j, fj,
    initial_x::AbstractArray{T},
    method::TM = NewtonTrustRegion(),
    options = Options(T); inplace = true) where {T, TM<:AbstractNLSolver}

    if inplace
        df = OnceDifferentiable(f, j, fj, initial_x, initial_x)
    else
        df = OnceDifferentiable(not_in_place(f, j, fj)..., initial_x, initial_x)
    end

    nlsolve(df, initial_x, method, options)
end