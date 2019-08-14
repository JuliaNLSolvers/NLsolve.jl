function nlsolve(df::TDF,
                 initial_x::AbstractArray{T};
                 method::Symbol = :trust_region,
                 xtol::Real = zero(real(T)),
                 ftol::Real = convert(real(T), 1e-8),
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch = LineSearches.Static(),
                 linsolve=(x, A, b) -> copyto!(x, A\b),
                 factor::Real = one(real(T)),
                 autoscale::Bool = true,
                 m::Integer = 10,
                 beta::Real = 1,
                 aa_start::Integer = 1,
                 droptol::Real = convert(real(T), 1e10)) where {T, TDF <: Union{NonDifferentiable, OnceDifferentiable}}
    if show_trace
        @printf "Iter     f(x) inf-norm    Step 2-norm \n"
        @printf "------   --------------   --------------\n"
    end
    if method == :newton
        newton(df, initial_x, xtol, ftol, iterations,
               store_trace, show_trace, extended_trace, linesearch; linsolve=linsolve)
    elseif method == :trust_region
        trust_region(df, initial_x, xtol, ftol, iterations,
                     store_trace, show_trace, extended_trace, factor,
                     autoscale)
    elseif method == :anderson
        anderson(df, initial_x, xtol, ftol, iterations,
                 store_trace, show_trace, extended_trace, m, beta, aa_start, droptol)
    elseif method == :broyden
        broyden(df, initial_x, xtol, ftol, iterations,
                store_trace, show_trace, extended_trace, linesearch)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function nlsolve(f,
                 initial_x::AbstractArray{T};
                 method::Symbol = :trust_region,
                 xtol::Real = zero(real(T)),
                 ftol::Real = convert(real(T), 1e-8),
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch = LineSearches.Static(),
                 factor::Real = one(real(T)),
                 autoscale::Bool = true,
                 m::Integer = 10,
                 beta::Real = 1,
                 aa_start::Integer = 1,
                 droptol::Real = convert(real(T), 1e10),
                 autodiff = :central,
                 linsolve=(x, A, b) -> copyto!(x, A\b),
                 inplace = !applicable(f, initial_x)) where T
    if typeof(f) <: Union{InplaceObjective, NotInplaceObjective}
        if !(method in (:anderson, :broyden))
            df = OnceDifferentiable(f, initial_x, similar(initial_x); autodiff=autodiff, inplace=inplace)
        else
            df = NonDifferentiable(f, initial_x, similar(initial_x); inplace=inplace)
        end
    else
        if !(method in (:anderson, :broyden))
            df = OnceDifferentiable(f, initial_x, similar(initial_x); autodiff=autodiff, inplace=inplace)
        else
            df = NonDifferentiable(f, initial_x, similar(initial_x); inplace=inplace)
        end
    end

    nlsolve(df,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch = linesearch, factor = factor, autoscale = autoscale,
            m = m, beta = beta, aa_start = aa_start, droptol = droptol, linsolve=linsolve)
end


function nlsolve(f,
                j,
                initial_x::AbstractArray{T};
                method::Symbol = :trust_region,
                xtol::Real = zero(real(T)),
                ftol::Real = convert(real(T), 1e-8),
                iterations::Integer = 1_000,
                store_trace::Bool = false,
                show_trace::Bool = false,
                extended_trace::Bool = false,
                linesearch = LineSearches.Static(),
                factor::Real = one(real(T)),
                autoscale::Bool = true,
                m::Integer = 10,
                beta::Real = 1,
                aa_start::Integer = 1,
                droptol::Real = convert(real(T), 1e10),
                inplace = !applicable(f, initial_x),
                linsolve=(x, A, b) -> copyto!(x, A\b)) where T
    if inplace
        df = OnceDifferentiable(f, j, initial_x, similar(initial_x))
    else
        df = OnceDifferentiable(not_in_place(f, j)..., initial_x, similar(initial_x))
    end
    nlsolve(df,
    initial_x, method = method, xtol = xtol, ftol = ftol,
    iterations = iterations, store_trace = store_trace,
    show_trace = show_trace, extended_trace = extended_trace,
    linesearch = linesearch, factor = factor, autoscale = autoscale,
    m = m, beta = beta, aa_start = aa_start, droptol = droptol, linsolve=linsolve)
end

function nlsolve(f,
                j,
                fj,
                initial_x::AbstractArray{T};
                method::Symbol = :trust_region,
                xtol::Real = zero(real(T)),
                ftol::Real = convert(real(T), 1e-8),
                iterations::Integer = 1_000,
                store_trace::Bool = false,
                show_trace::Bool = false,
                extended_trace::Bool = false,
                linesearch = LineSearches.Static(),
                factor::Real = one(real(T)),
                autoscale::Bool = true,
                m::Integer = 10,
                beta::Real = 1,
                aa_start::Integer = 1,
                droptol::Real = convert(real(T), 1e10),
                inplace = !applicable(f, initial_x),
                linsolve=(x, A, b) -> copyto!(x, A\b)) where T
    if inplace
        df = OnceDifferentiable(f, j, fj, initial_x, similar(initial_x))
    else
        df = OnceDifferentiable(not_in_place(f, j, fj)..., initial_x, similar(initial_x))
    end
    nlsolve(df,
    initial_x, method = method, xtol = xtol, ftol = ftol,
    iterations = iterations, store_trace = store_trace,
    show_trace = show_trace, extended_trace = extended_trace,
    linesearch = linesearch, factor = factor, autoscale = autoscale,
    m = m, beta = beta, aa_start = aa_start, droptol = droptol, linsolve=linsolve)
end
