using NLsolve

tests = ["2by2.jl",
         "singular.jl",
         "finite_difference.jl",
         "minpack.jl",
         "iface.jl",
         "already_converged.jl",
         "autodiff.jl",
         "josephy.jl"]

println("Running tests:")

for t in tests
    println(" * $(t)")
    include(t)
end
