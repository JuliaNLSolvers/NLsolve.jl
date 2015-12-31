VERSION >= v"0.4.0-dev+6521" && __precompile__()

module NLsolve

using Distances
using Optim
using ForwardDiff

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

include("differentiable_functions.jl")
include("solver_state_results.jl")
include("nlsolve_func_defs.jl")
include("mcp_func_defs.jl")
include("utils.jl")
include("newton.jl")
include("trust_region.jl")
include("autodiff.jl")
include("mcp.jl")

end # module
