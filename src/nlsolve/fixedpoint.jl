
# Method for function with no Jacobian.
function fixedpoint(f,
    initial_x::AbstractArray{T};
    method::Symbol = :anderson,
    xtol::Real = zero(T),
    ftol::Real = convert(T,1e-8),
    iterations::Integer = 1_000,
    store_trace::Bool = false,
    show_trace::Bool = false,
    extended_trace::Bool = false,
    linesearch = LineSearches.Static(),
    factor::Real = one(T),
    autoscale::Bool = true,
    m::Integer = 5,
    beta::Real = 1,
    droptol::Real = 0,
    autodiff::Symbol = :central,
    inplace::Bool = !applicable(f, initial_x)) where T
    # Check for weird case. (Causes to hang for now)
    # typeof(f) <: Union{InplaceObjective, NotInplaceObjective} ? error("Union{InplaceObjective, NotInplaceObjective} Case") : true;
    # Wrapping
    if inplace
        function g!(out, x)
            f(out, x);
            out .-= x;
        end
        dg = OnceDifferentiable(g!, initial_x, copy(initial_x), autodiff)
    else
        g(x) = f(x) - x;
        dg = OnceDifferentiable(g, initial_x, copy(initial_x); autodiff = autodiff, inplace = inplace)
    end

    return nlsolve(dg,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch = linesearch, factor = factor, autoscale = autoscale,
            m = m, beta = beta, droptol = droptol)
end
