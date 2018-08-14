__precompile__()

module NLsolve

using Distances
using NLSolversBase
using LineSearches
using ForwardDiff
using DiffEqDiffTools
using LinearAlgebra
using Printf

import Base.show,
       Base.push!,
       Base.getindex,
       Base.setindex!

import NLSolversBase: OnceDifferentiable, InplaceObjective, NotInplaceObjective,
       only_fj, only_fj!

using Reexport
@reexport using LineSearches
using LinearAlgebra

export OnceDifferentiable,
       n_ary,
       nlsolve,
       mcpsolve,
       converged,
       only_fj,
       only_fj!,
       fixedpoint

abstract type AbstractSolver end
abstract type AbstractSolverCache end

"""
    cache(::AbstractSolver, df::OnceDifferentiable)

Return a solver cache that can be used to reduce memory allocations on
successive calls to `nlsolve` with the same problem structure.
"""
function cache end

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
include("nlsolve/fixedpoint.jl")

end # module
