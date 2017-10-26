# taken from https://github.com/JuliaNLSolvers/NLsolve.jl/issues/121
@testset "issue 121" begin
function f(x)
    F = Array{Float64}(length(x))

    F[1] = 5 .* x[1] - x[2].^2
    F[2] = 4 .* x[2] - x[1]

    return F
end

function j(x)
    J = Array{Float64}(length(x),length(x))

    J[1,1] = 5
    J[1,2] = -2.*x[2]
    J[2,1] = -1
    J[2,2] = 4

    return J
end

x0 = [3.0, 12.0]

f_x0 = f(x0)
j_x0 = j(x0)

F = similar(x0)
not_in_place(f)(F, x0)
J = similar(x0, length(x0), length(x0))
not_in_place(j)(J, x0)
@test f_x0 == F
@test j_x0 == J
end
