using NLsolve
using Test
using NLSolversBase
using LineSearches
using LinearAlgebra
using SparseArrays
using Printf
using IterativeSolvers
using Random
add_jl(x) = endswith(x, ".jl") ? x : x*".jl"

if length(ARGS) > 0
    tests = map(add_jl, ARGS)
else
    tests = ["2by2.jl",
             "linsolve.jl",
             "minpack.jl",
             "singular.jl",
             "finite_difference.jl",
             "iface.jl",
             "incomplete.jl",
             "already_converged.jl",
             "autodiff.jl",
             "josephy.jl",
             "difficult_mcp.jl",
             "sparse.jl",
             "throws.jl",
             "f_g_counts.jl",
             "no_linesearch.jl",
             "abstractarray.jl",
             "complex.jl",
             "interface/caches.jl",
             "fixedpoint/fixedpoint.jl"]
end

println("Running tests:")

for test in tests
    include(test)
end
