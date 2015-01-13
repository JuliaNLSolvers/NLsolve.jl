module NLsolve

using Optim

import Base.show,
       Base.push!,
       Base.getindex,
       Base.setindex!

import Calculus.finite_difference_jacobian!

export DifferentiableMultivariateFunction,
       only_f!_and_fg!,
       only_fg!,
       not_in_place,
       n_ary,
       DifferentiableSparseMultivariateFunction,
       nlsolve,
       mcpsolve,
       converged

abstract AbstractDifferentiableMultivariateFunction

immutable DifferentiableMultivariateFunction <: AbstractDifferentiableMultivariateFunction
    f!::Function
    g!::Function
    fg!::Function
end

alloc_jacobian(df::DifferentiableMultivariateFunction, T::Type, n::Integer) = Array(T, n, n)

function DifferentiableMultivariateFunction(f!::Function, g!::Function)
    function fg!(x::Vector, fx::Vector, gx::Array)
        f!(x, fx)
        g!(x, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function DifferentiableMultivariateFunction(f!::Function)
    function fg!(x::Vector, fx::Vector, gx::Array)
        f!(x, fx)
        function f(x::Vector)
            fx = similar(x)
            f!(x, fx)
            return fx
        end
        finite_difference_jacobian!(f, x, fx, gx)
    end
    function g!(x::Vector, gx::Array)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

# Helper for the case where only f! and fg! are available
function only_f!_and_fg!(f!::Function, fg!::Function)
    function g!(x::Vector, gx::Array)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

# Helper for the case where only fg! is available
function only_fg!(fg!::Function)
    function f!(x::Vector, fx::Vector)
        gx = Array(Float64, length(x), length(x))
        fg!(x, fx, gx)
    end
    function g!(x::Vector, gx::Array)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

# Helpers for functions that do not modify arguments in place but return
# values and jacobian
function not_in_place(f::Function)
    function f!(x::Vector, fx::Vector)
        copy!(fx, f(x))
    end
    DifferentiableMultivariateFunction(f!)
end

function not_in_place(f::Function, g::Function)
    function f!(x::Vector, fx::Vector)
        copy!(fx, f(x))
    end
    function g!(x::Vector, gx::Array)
        copy!(gx, g(x))
    end
    DifferentiableMultivariateFunction(f!, g!)
end

function not_in_place(f::Function, g::Function, fg::Function)
    function f!(x::Vector, fx::Vector)
        copy!(fx, f(x))
    end
    function g!(x::Vector, gx::Array)
        copy!(gx, g(x))
    end
    function fg!(x::Vector, fx::Vector, gx::Array)
        (fvec, fjac) = fg(x)
        copy!(fx, fvec)
        copy!(gx, fjac)
    end
    DifferentiableMultivariateFunction(f!, g!, fg!)
end

# Helper for functions that take several scalar arguments and return a tuple
function n_ary(f::Function)
    function f!(x::Vector, fx::Vector)
        copy!(fx, [ f(x...)... ])
    end
    DifferentiableMultivariateFunction(f!)
end

# For sparse Jacobians
immutable DifferentiableSparseMultivariateFunction <: AbstractDifferentiableMultivariateFunction
    f!::Function
    g!::Function
    fg!::Function
end

alloc_jacobian(df::DifferentiableSparseMultivariateFunction, T::Type, n::Integer) = spzeros(T, n, n)

function DifferentiableSparseMultivariateFunction(f!::Function, g!::Function)
    function fg!(x::Vector, fx::Vector, gx::SparseMatrixCSC)
        f!(x, fx)
        g!(x, gx)
    end
    return DifferentiableSparseMultivariateFunction(f!, g!, fg!)
end

immutable SolverState
    iteration::Int
    fnorm::Float64
    stepnorm::Float64
    metadata::Dict
end

function SolverState(i::Integer, fnorm::Real)
    SolverState(int(i), float64(fnorm), NaN, Dict())
end

function SolverState(i::Integer, fnorm::Real, stepnorm::Real)
    SolverState(int(i), float64(fnorm), float64(stepnorm), Dict())
end

immutable SolverTrace
    states::Vector{SolverState}
end

SolverTrace() = SolverTrace(Array(SolverState, 0))

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

type SolverResults{T}
    method::ASCIIString
    initial_x::Vector{T}
    zero::Vector{T}
    residual_norm::Float64
    iterations::Int
    x_converged::Bool
    xtol::Float64
    f_converged::Bool
    ftol::Float64
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


function assess_convergence(x::Vector,
                            x_previous::Vector,
                            f::Vector,
                            xtol::Real,
                            ftol::Real)
    x_converged, f_converged = false, false

    if norm(x - x_previous, Inf) < xtol
        x_converged = true
    end

    if norm(f, Inf) < ftol
        f_converged = true
    end

    converged = x_converged || f_converged

    return x_converged, f_converged, converged
end

include("newton.jl")
include("trust_region.jl")
include("autodiff.jl")
include("mcp.jl")

function nlsolve(df::AbstractDifferentiableMultivariateFunction,
                 initial_x::Vector;
                 method::Symbol = :trust_region,
                 xtol::Real = 0.0,
                 ftol::Real = 1e-8,
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch!::Function = Optim.backtracking_linesearch!,
                 factor::Real = 1.0,
                 autoscale::Bool = true)
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
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

function nlsolve(f!::Function,
                 g!::Function,
                 initial_x::Vector;
                 method::Symbol = :trust_region,
                 xtol::Real = 0.0,
                 ftol::Real = 1e-8,
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch!::Function = Optim.backtracking_linesearch!,
                 factor::Real = 1.0,
                 autoscale::Bool = true)
    nlsolve(DifferentiableMultivariateFunction(f!, g!),
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end

function nlsolve(f!::Function,
                 initial_x::Vector;
                 method::Symbol = :trust_region,
                 xtol::Real = 0.0,
                 ftol::Real = 1e-8,
                 iterations::Integer = 1_000,
                 store_trace::Bool = false,
                 show_trace::Bool = false,
                 extended_trace::Bool = false,
                 linesearch!::Function = Optim.backtracking_linesearch!,
                 factor::Real = 1.0,
                 autoscale::Bool = true,
                 autodiff::Bool = false)
    if !autodiff
        df = DifferentiableMultivariateFunction(f!)
    else
        df = NLsolve.autodiff(f!, eltype(initial_x), length(initial_x), length(initial_x))
    end
    nlsolve(df,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end


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

function mcpsolve(df::AbstractDifferentiableMultivariateFunction,
                  lower::Vector,
                  upper::Vector,
                  initial_x::Vector;
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = 0.0,
                  ftol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = Optim.backtracking_linesearch!,
                  factor::Real = 1.0,
                  autoscale::Bool = true)
    @reformulate df
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end

function mcpsolve(f!::Function,
                  g!::Function,
                  lower::Vector,
                  upper::Vector,
                  initial_x::Vector;
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = 0.0,
                  ftol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = Optim.backtracking_linesearch!,
                  factor::Real = 1.0,
                  autoscale::Bool = true)
    @reformulate DifferentiableMultivariateFunction(f!, g!)
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end

function mcpsolve(f!::Function,
                  lower::Vector,
                  upper::Vector,
                  initial_x::Vector;
                  method::Symbol = :trust_region,
                  reformulation::Symbol = :smooth,
                  xtol::Real = 0.0,
                  ftol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  linesearch!::Function = Optim.backtracking_linesearch!,
                  factor::Real = 1.0,
                  autoscale::Bool = true,
                  autodiff::Bool = false)
    if !autodiff
        df = DifferentiableMultivariateFunction(f!)
    else
        df = NLsolve.autodiff(f!, eltype(initial_x), length(initial_x), length(initial_x))
    end
    @reformulate df
    nlsolve(rf,
            initial_x, method = method, xtol = xtol, ftol = ftol,
            iterations = iterations, store_trace = store_trace,
            show_trace = show_trace, extended_trace = extended_trace,
            linesearch! = linesearch!, factor = factor, autoscale = autoscale)
end

end # module
