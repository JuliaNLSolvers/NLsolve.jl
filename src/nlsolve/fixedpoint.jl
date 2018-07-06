
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
    linesearch = LineSearches.NewStatic(),
    factor::Real = one(T),
    autoscale::Bool = true,
    m::Integer = 0,
    beta::Real = 1.0,
    autodiff::Symbol = :central,
    inplace::Bool = true) where T
    # Check for weird case. (Causes to hang for now)
    # typeof(f) <: Union{InplaceObjective, NotInplaceObjective} ? error("Union{InplaceObjective, NotInplaceObjective} Case") : true; 
    # Wrapping 
    if inplace
        function g!(out, x)
            f(out, x); 
            out .-= x;
        end 
        dg = OnceDifferentiable(g!, initial_x, initial_x, autodiff) 
    else 
        g(x) = f(x) - x;
        dg = OnceDifferentiable(not_in_place(g), initial_x, initial_x, autodiff)
    end 

    return nlsolve(dg,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch = linesearch, factor = factor, autoscale = autoscale,
            m = m, beta = beta)
end 




