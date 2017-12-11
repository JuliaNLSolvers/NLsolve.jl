# Verify that we handle correctly the case where the starting point is a zero
@testset "already converged" begin
function f!(fvec, x)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

df = OnceDifferentiable(f!, rand(2), rand(2))

r = nlsolve(df, [ 0.; 1.], method = :trust_region)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8
@test r.iterations == 0

r = nlsolve(df, [ 0.; 1.], method = :newton)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8
@test r.iterations == 0
end