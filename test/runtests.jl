using NLsolve
using DiffEqBase
using Test
using NLSolversBase
using LinearAlgebra
import Base.convert
using SparseArrays
using Printf

add_jl(x) = endswith(x, ".jl") ? x : x*".jl"

mutable struct WrappedArray{T,N} <: DEDataArray{T,N}
    x::Array{T,N}
end

Base.reshape(A::WrappedArray, dims::Int...) = WrappedArray(reshape(A.x, dims...))
Base.convert(A::Type{WrappedArray{T,N}}, B::Array{T,N}) where {T,N} = WrappedArray(copy(B))

if length(ARGS) > 0
    tests = map(add_jl, ARGS)
else
    tests = [# "2by2.jl",
             "singular.jl",
             "finite_difference.jl",
<<<<<<< HEAD
             "minpack.jl",
             "iface.jl",
             "incomplete.jl",
=======
            # "minpack.jl",
            # "iface.jl",
>>>>>>> comment out some tests + reorganized fixed-point
             "already_converged.jl",
             "autodiff.jl",
             "josephy.jl",
             "difficult_mcp.jl",
             "sparse.jl",
             "throws.jl",
             "f_g_counts.jl",
             "no_linesearch.jl",
             "abstractarray.jl",
             "fixedpoint.jl"]
end

println("Running tests:")

for test in tests
    include(test)
end
