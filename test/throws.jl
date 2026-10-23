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

# A residual count that differs from the number of unknowns gets a
# DimensionMismatch, not a BoundsError or a stale residual (#270, #291)
x0 = [0.3, 0.2]
f_tall(x) = [x[1] - 1, x[2] - 2, x[1] + x[2] - 3]
j_tall(x) = [1.0 0.0; 0.0 1.0; 1.0 1.0]
f_wide(x) = sin(x[1]) * sin(x[2])
for method in (:trust_region, :newton, :anderson, :broyden)
    @test_throws "returned an output of length 3, but length 2 was expected" nlsolve(f_tall, x0; method)
    @test_throws DimensionMismatch nlsolve(f_wide, x0; method)
end
@test_throws DimensionMismatch nlsolve(f_tall, x0; autodiff = AutoForwardDiff())
@test_throws DimensionMismatch nlsolve(f_tall, j_tall, x0)
@test_throws DimensionMismatch nlsolve(f_tall, j_tall, x -> (f_tall(x), j_tall(x)), x0)

function f_tall!(F, x)
    F[1] = x[1] - 1
    F[2] = x[2] - 2
    F[3] = x[1] + x[2] - 3
end
df_tall = OnceDifferentiable(f_tall!, x0, zeros(3))
for method in (:trust_region, :anderson, :broyden)
    @test_throws "needs one residual per unknown" nlsolve(df_tall, x0; method)
end
# Newton's J \ F step is a least-squares step, so :newton accepts a tall system
r = nlsolve(df_tall, x0; method = :newton)
@test converged(r)
@test r.zero ≈ [1.0, 2.0]

end # testset
