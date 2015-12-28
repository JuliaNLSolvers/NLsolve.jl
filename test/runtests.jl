using NLsolve
using Compat

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end



tests = ["2by2.jl",
         "singular.jl",
         "finite_difference.jl",
         "minpack.jl",
         "iface.jl",
         "already_converged.jl",
         "autodiff.jl",
         "josephy.jl",
         "difficult_mcp.jl",
         "sparse.jl"]

println("Running tests:")

for test in tests
    include(test)    
end
