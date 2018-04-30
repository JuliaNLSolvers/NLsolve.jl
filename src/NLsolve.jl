__precompile__()

module NLsolve

#using Distances
using NLSolversBase
using LineSearches
using ForwardDiff
using DiffEqDiffTools

import Parameters: @with_kw, @unpack
import Base.show,
       Base.push!,
       Base.getindex,
       Base.setindex!

import NLSolversBase: OnceDifferentiable, InplaceObjective, NotInplaceObjective,
       only_fj, only_fj!

export OnceDifferentiable,
       n_ary,
       nlsolve,
       mcpsolve,
       converged,
       only_fj,
       only_fj!

export Newton, NewtonTrustRegion, Anderson

abstract type AbstractSolverCache end
abstract type AbstractSolver end

"""
    Options(; kwargs...)

Construct an Options instance that holds options that are not specific to any
specific algorithm.

# Keyword arguments

- `x_abstol`: tolerance for largest absolute value of Δxₖ
- `f_abstol`: tolerance for largest absolute value of ΔFₖ
- `iterations`:  maximum number of iterations
- `store_trace`: store a trace of ΔFₖ, Δxₖ, and iteration number
- `show_trace`:  print the traced values after each iteration
- `extended_trace`: store algorithm specific values
- `autoscale`: automatically scale each step by the per-column-norm of ∇F
"""
@with_kw struct Options{T}
    x_abstol::T = 0.0
    f_tol::T = 0.0
    iterations::Int = 10^3
    store_trace::Bool = false
    show_trace::Bool = false
    extended_trace::Bool = false
    autoscale::Bool = true
end

"""
    IsFiniteException(indices)

Construct an instance of an exception type used to indicate that some elements
of a vector was not finite.
"""
struct IsFiniteException <: Exception
  indices::Vector{Int}
end

show(io::IO, e::IsFiniteException) = print(io,
  "During the resolution of the non-linear system, the evaluation" *
  " of the following equation(s) resulted in a non-finite number: $(e.indices)")

include("objectives/helpers.jl")

include("solvers/newton.jl")
include("solvers/trust_region.jl")
include("solvers/anderson.jl")
include("solvers/mcp_func_defs.jl")
include("solvers/mcp.jl")

include("nlsolve/solver_state_results.jl")
include("nlsolve/nlsolve.jl")
include("nlsolve/utils.jl")

end # module
