import NLsolve: IsFiniteException

@testset "throws" begin

function f_inf!(F, x)
    copyto!(F, x)
    F[1] = Inf
    return F
end

function f_nan!(F, x)
    copyto!(F, x)
    F[1] = NaN
    return F
end

@test_throws IsFiniteException nlsolve(f_inf!, [ -0.5; 1.4], method = :trust_region, autodiff=AutoForwardDiff())
@test_throws IsFiniteException nlsolve(f_inf!, [ -0.5; 1.4], method = :newton, autodiff=AutoForwardDiff())

@test_throws IsFiniteException nlsolve(f_nan!, [ -0.5; 1.4], method = :trust_region, autodiff=AutoForwardDiff())
@test_throws IsFiniteException nlsolve(f_nan!, [ -0.5; 1.4], method = :newton, autodiff=AutoForwardDiff())

# autodiff must be an ADTypes backend; anything else gets an ArgumentError
# naming the replacement, not a MethodError (#268)
function f_ok!(F, x)
    F[1] = x[1]^2 - 2
    F[2] = x[2]^2 - 3
end
@test_throws ArgumentError nlsolve(f_ok!, [1.0, 1.0], autodiff = :forward)
@test_throws ArgumentError nlsolve(f_ok!, [1.0, 1.0], autodiff = :central)
@test_throws ArgumentError nlsolve(f_ok!, [1.0, 1.0], autodiff = true)
@test_throws ArgumentError fixedpoint(f_ok!, [1.0, 1.0], autodiff = true)
@test_throws ArgumentError mcpsolve(f_ok!, [0.0, 0.0], [10.0, 10.0], [1.0, 1.0], autodiff = true)

end # testset
