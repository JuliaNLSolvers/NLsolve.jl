using NLsolve
using DiffEqBase
using Base.Test

add_jl(x) = endswith(x, ".jl") ? x : x*".jl"

type WrappedArray{T,N} <: DEDataArray{T,N}
    x::Array{T,N}
end

Base.A_ldiv_B!(F::Factorization, B::WrappedArray) = A_ldiv_B!(F, B.x)
Base.reshape(A::WrappedArray, dims::Int...) = WrappedArray(reshape(A.x, dims...))

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
             "no_linesearch.jl",
             "abstractarray.jl"]
end

println("Running tests:")

for test in tests
    include(test)
end
