# Test sparse Jacobian implementation (even though the Jacobian of the function
# is actually not sparse...)

@testset "sparse" begin

function f_sparse!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

function g_sparse!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end

df = OnceDifferentiable(f_sparse!, g_sparse!, rand(2), spzeros(2, 2), rand(2))

# Test trust region
r = nlsolve(df, [ -0.5; 1.4], method = :trust_region, autoscale = true)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8

# Test Newton
r = nlsolve(df, [ -0.5; 1.4], method = :newton, linesearch! = LineSearches.BackTracking(), ftol = 1e-6)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-6

# Test MCP solver with smooth reformulation
r = mcpsolve(df, [-Inf;-Inf], [Inf; Inf], [-0.5; 1.4], reformulation = :smooth)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8

# Test MCP solver with minmax reformulation
r = mcpsolve(df, [-Inf;-Inf], [Inf; Inf], [-0.5; 1.4], reformulation = :minmax)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8

# Test given sparse
df = OnceDifferentiable(f_sparse!, g_sparse!, rand(2), spzeros(2, 2), rand(2))
r = nlsolve(df, [ -0.5; 1.4], method = :trust_region, autoscale = true)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8

end
