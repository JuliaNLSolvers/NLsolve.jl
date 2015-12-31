import NLsolve: IsFiniteException

@testset "throws" begin

function f_inf!(x, fx)
    copy!(fx, x)
    fx[1] = Inf
    return fx
end

function f_nan!(x, fx)
    copy!(fx, x)
    fx[1] = NaN
    return fx
end

@test_throws IsFiniteException nlsolve(f_inf!, [ -0.5; 1.4], method = :trust_region, autodiff=true)
@test_throws IsFiniteException nlsolve(f_inf!, [ -0.5; 1.4], method = :newton, autodiff=true)

@test_throws IsFiniteException nlsolve(f_nan!, [ -0.5; 1.4], method = :trust_region, autodiff=true)
@test_throws IsFiniteException nlsolve(f_nan!, [ -0.5; 1.4], method = :newton, autodiff=true)

end # testset