# Example from Nocedal & Wright, p. 281
# Used to test all the different algorithms
@testset "2by2" begin


function f_2by2!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

function g_2by2!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end

df = OnceDifferentiable(f_2by2!, g_2by2!, [ -0.5; 1.4], [ -0.5; 1.4])

# Test trust region
r = nlsolve(df, [ -0.5; 1.4], method = :trust_region, autoscale = true)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-7
r = nlsolve(df, [ -0.5; 1.4], method = :trust_region, autoscale = false)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-7

df32 = OnceDifferentiable(f_2by2!, g_2by2!, [ -0.5f0; 1.4f0], [ -0.5f0; 1.4f0])
r = nlsolve(df32, [ -0.5f0; 1.4f0], method = :trust_region, autoscale = true)
@test eltype(r.zero) == Float32
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-7
r = nlsolve(df32, [ -0.5f0; 1.4f0], method = :trust_region, autoscale = false)
@test eltype(r.zero) == Float32
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-7

# Test Newton
r = nlsolve(df, [ -0.5; 1.4], method = :newton, linesearch = LineSearches.BackTracking(), ftol = 1e-6)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-6
r = nlsolve(df32, [ -0.5f0; 1.4f0], method = :newton, linesearch = LineSearches.BackTracking(), ftol = 1e-3)
@test eltype(r.zero) == Float32
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-6
r = nlsolve(df, [ -0.5; 1.4], method = :newton, linesearch = LineSearches.HagerZhang(), ftol = 1e-6)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-6
r = nlsolve(df, [ -0.5; 1.4], method = :newton, linesearch = LineSearches.StrongWolfe(), ftol = 1e-6)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-6

# test local convergence of Anderson: close to a fixed-point and with
# a small beta, f should be almost affine, in which case Anderson is
# equivalent to GMRES and should converge
r = nlsolve(df, [ 0.01; .99]; method = :anderson, m = 10, beta=.01)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8
end
