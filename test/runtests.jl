using NLsolve

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

add_jl(x) = endswith(x, ".jl") ? x : x*".jl"

if length(ARGS) > 0
    tests = map(add_jl, ARGS)
else
    tests = ["2by2.jl",
             "singular.jl",
             "finite_difference.jl",
             "minpack.jl",
             "iface.jl",
             "already_converged.jl",
             "autodiff.jl",
             "josephy.jl",
             "difficult_mcp.jl",
             "sparse.jl",
             "throws.jl",
             "f_g_counts.jl",
             "no_linesearch.jl"]
end

println("Running tests:")

for test in tests
    include(test)
end
