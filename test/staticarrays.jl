# MArray inputs (#254, #228)

@testset "MArray" begin

function f_ms!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

function j_ms!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end

solution = [0.0, 1.0]

# Float32 with ForwardDiff (#254)
r = nlsolve(f_ms!, MVector{2, Float32}(0.1, 1.2); autodiff = AutoForwardDiff())
@test converged(r)
@test r.zero isa MVector{2, Float32}
@test norm(r.zero - solution) < 1e-5

# Analytic Jacobian (#228)
r = nlsolve(f_ms!, j_ms!, MVector(0.1, 1.2))
@test converged(r)
@test r.zero isa MVector{2, Float64}
@test norm(r.zero - solution) < 1e-8

for method in (:trust_region, :newton, :anderson, :broyden)
    r = nlsolve(f_ms!, MVector(0.1, 1.2); method)
    @test converged(r)
    @test r.zero isa MVector{2, Float64}
    @test norm(r.zero - solution) < 1e-8
end

end
