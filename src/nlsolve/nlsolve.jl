function nlsolve(df::Union{NonDifferentiable, OnceDifferentiable},
                 initial_x::AbstractArray;
                 method::Symbol = :trust_region,
                 xtol::Real = zero(real(eltype(initial_x))),
                 ftol::Real = convert(real(eltype(initial_x)), 1e-8),
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch = LineSearches.Static(),
                 linsolve=(x, A, b) -> copyto!(x, A\b),
                 factor::Real = one(real(eltype(initial_x))),
                 autoscale::Bool = true,
                 m::Integer = 10,
                 beta::Real = 1,
                 aa_start::Integer = 1,
                 droptol::Real = convert(real(eltype(initial_x)), 1e10))
    if show_trace
        @printf "Iter     f(x) inf-norm    Step 2-norm \n"
        @printf "------   --------------   --------------\n"
    end
    # Newton's J \ F step is a least-squares step, so only the other methods need a square system
    if method in (:trust_region, :anderson, :broyden) && length(value(df)) != length(initial_x)
        throw(DimensionMismatch("method = :$method needs one residual per unknown, but the residual has length $(length(value(df))) and initial_x has length $(length(initial_x)). For least squares, see LsqFit.jl."))
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
                 initial_x::AbstractArray;
                 method::Symbol = :trust_region,
                 autodiff = DEFAULT_AUTODIFF,
                 inplace = !applicable(f, initial_x),
                 kwargs...)
    f! = inplace ? f : not_in_place(f)
    if method in (:anderson, :broyden)
        df = NonDifferentiable(f!, initial_x, copy(initial_x))
    else
        df = OnceDifferentiable(f!, initial_x, copy(initial_x); autodiff=check_autodiff(autodiff))
    end

    nlsolve(df, initial_x; method = method, kwargs...)
end


function nlsolve(f,
                 j,
                 initial_x::AbstractArray;
                 inplace = !applicable(f, initial_x),
                 kwargs...)
    if inplace
        df = OnceDifferentiable(f, j, initial_x, copy(initial_x))
    else
        df = OnceDifferentiable(not_in_place(f, j)..., initial_x, copy(initial_x))
    end

    nlsolve(df, initial_x; kwargs...)
end

function nlsolve(f,
                 j,
                 fj,
                 initial_x::AbstractArray;
                 inplace = !applicable(f, initial_x),
                 kwargs...)
    if inplace
        df = OnceDifferentiable(f, j, fj, initial_x, copy(initial_x))
    else
        df = OnceDifferentiable(not_in_place(f, j, fj)..., initial_x, copy(initial_x))
    end

    nlsolve(df, initial_x; kwargs...)
end
