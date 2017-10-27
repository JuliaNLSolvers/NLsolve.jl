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

export DifferentiableVector,
       only_f!_and_fj!,
       only_fj!,
       not_in_place,
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
df_path = "differentiable_vectors/"
include(df_path*"differentiable_functions.jl")
include(df_path*"interface.jl")
include(df_path*"autodiff.jl")
include(df_path*"helpers.jl")
s_path = "solvers/"
include(s_path*"newton.jl")
include(s_path*"trust_region.jl")
include(s_path*"anderson.jl")
include(s_path*"mcp_func_defs.jl")
include(s_path*"mcp.jl")
nls_path = "nlsolve/"
include(nls_path*"solver_state_results.jl")
include(nls_path*"nlsolve.jl")
include(nls_path*"utils.jl")

end # module
