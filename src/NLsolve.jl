__precompile__()

module NLsolve

using Distances
using NLSolversBase
using LineSearches
using ForwardDiff

import Base.show,
       Base.push!,
       Base.getindex,
       Base.setindex!

import Calculus.finite_difference_jacobian!
import NLSolversBase: OnceDifferentiable
export OnceDifferentiable,
       n_ary,
       nlsolve,
       mcpsolve,
       converged

struct IsFiniteException <: Exception
  indices::Vector{Int}
end
show(io::IO, e::IsFiniteException) = print(io,
  "During the resolution of the non-linear system, the evaluation" *
  " of the following equation(s) resulted in a non-finite number: $(e.indices)")

include("differentiable_vectors/autodiff.jl")
include("differentiable_vectors/helpers.jl")

include("solvers/newton.jl")
include("solvers/trust_region.jl")
include("solvers/anderson.jl")
include("solvers/mcp_func_defs.jl")
include("solvers/mcp.jl")

include("nlsolve/solver_state_results.jl")
include("nlsolve/nlsolve.jl")
include("nlsolve/utils.jl")

end # module
