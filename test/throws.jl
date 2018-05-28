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

@test_throws IsFiniteException nlsolve(f_inf!, [ -0.5; 1.4], method = :trust_region, autodiff=true)
@test_throws IsFiniteException nlsolve(f_inf!, [ -0.5; 1.4], method = :newton, autodiff=true)

@test_throws IsFiniteException nlsolve(f_nan!, [ -0.5; 1.4], method = :trust_region, autodiff=true)
@test_throws IsFiniteException nlsolve(f_nan!, [ -0.5; 1.4], method = :newton, autodiff=true)

end # testset
