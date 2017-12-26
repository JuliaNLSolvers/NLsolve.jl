__precompile__()

module NLsolve

using Distances
using NLSolversBase
using LineSearches
using ForwardDiff
using DiffEqDiffTools

import Base.show,
       Base.push!,
       Base.getindex,
       Base.setindex!

import NLSolversBase: OnceDifferentiable

export OnceDifferentiable,
       n_ary,
       nlsolve,
       mcpsolve,
       converged,
       Anderson,
       Newton,
       NewtonTrustRegion

abstract type AbstractNLSolver end

struct IsFiniteException <: Exception
  indices::Vector{Int}
end
show(io::IO, e::IsFiniteException) = print(io,
  "During the resolution of the non-linear system, the evaluation" *
  " of the following equation(s) resulted in a non-finite number: $(e.indices)")

<<<<<<< HEAD
=======
struct Options
    xtol
    ftol
    iterations
    show_trace
    store_trace
    extended_trace
end
Options(T) = Options(T(0.0), T(1e-8), 1000, false, false, false)
Options(;xtol=0.0, ftol=1e-8, iterations=1000, show_trace=false, store_trace=false, extended_trace=false) =
    Options(xtol, ftol, iterations, show_trace, store_trace, extended_trace)

>>>>>>> 311a942... Change to DiffEqDiffTools from Calculus. (#129)
include("objectives/autodiff.jl")
include("objectives/helpers.jl")

include("solvers/newton.jl")
include("solvers/trust_region.jl")
include("solvers/anderson.jl")
include("solvers/mcp_func_defs.jl")
include("solvers/mcp.jl")

include("nlsolve/solver_state_results.jl")
include("nlsolve/nlsolve.jl")
include("nlsolve/nlsolve_legacy.jl")
include("nlsolve/utils.jl")

end # module
