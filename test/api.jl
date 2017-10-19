# taken from https://github.com/JuliaNLSolvers/NLsolve.jl/issues/121

function f(x)
    eq = Array{Float64}(length(x))

    eq[1] = 5 .* x[1] - x[2].^2
    eq[2] = 4 .* x[2] - x[1]

    return eq
end

function J(x)
    jmat = Array{Float64}(length(x),length(x))

    jmat[1,1] = 5
    jmat[1,2] = -2.*x[2]
    jmat[2,1] = -1
    jmat[2,2] = 4

    return jmat
end

x0 = [3.0, 12.0]
