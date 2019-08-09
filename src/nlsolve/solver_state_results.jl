struct SolverState{T}
    iteration::Int
    fnorm::T
    stepnorm::T
    metadata::Dict
end

function SolverState(i, fnorm)
    SolverState(Int(i), fnorm, oftype(fnorm, NaN), Dict())
end

function SolverState(i, fnorm, stepnorm)
    SolverState(Int(i), fnorm, stepnorm, Dict())
end

struct SolverTrace
    states::Vector{SolverState}
end

SolverTrace() = SolverTrace(Array{SolverState}(undef, 0))

function Base.show(io::IO, t::SolverState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.fnorm t.stepnorm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

Base.push!(t::SolverTrace, s::SolverState) = push!(t.states, s)

Base.getindex(t::SolverTrace, i::Integer) = getindex(t.states, i)

function Base.setindex!(t::SolverTrace,
                        s::SolverState,
                        i::Integer)
    setindex!(t.states, s, i)
end

function Base.show(io::IO, t::SolverTrace)
    @printf io "Iter     f(x) inf-norm    Step 2-norm \n"
    @printf io "------   --------------   --------------\n"
    for state in t.states
        show(io, state)
    end
    return
end

function update!(tr::SolverTrace,
                 iteration::Integer,
                 fnorm::Real,
                 stepnorm::Real,
                 dt::Dict,
                 store_trace::Bool,
                 show_trace::Bool)
    ss = SolverState(iteration, fnorm, stepnorm, dt)
    if store_trace
        push!(tr, ss)
    end
    if show_trace
        show(ss)
    end
    return
end

mutable struct SolverResults{rT<:Real,T<:Union{rT,Complex{rT}},I<:AbstractArray{T},Z<:AbstractArray{T}}
    method::String
    initial_x::I
    zero::Z
    residual_norm::rT
    iterations::Int
    x_converged::Bool
    xtol::rT
    f_converged::Bool
    ftol::rT
    trace::SolverTrace
    f_calls::Int
    g_calls::Int
end

function converged(r::SolverResults)
    return r.x_converged || r.f_converged
end

function Base.show(io::IO, r::SolverResults)
    @printf io "Results of Nonlinear Solver Algorithm\n"
    @printf io " * Algorithm: %s\n" r.method
    @printf io " * Starting Point: %s\n" string(r.initial_x)
    @printf io " * Zero: %s\n" string(r.zero)
    @printf io " * Inf-norm of residuals: %f\n" r.residual_norm
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * |x - x'| < %.1e: %s\n" r.xtol r.x_converged
    @printf io "   * |f(x)| < %.1e: %s\n" r.ftol r.f_converged
    @printf io " * Function Calls (f): %d\n" r.f_calls
    @printf io " * Jacobian Calls (df/dx): %d" r.g_calls
    return
end
