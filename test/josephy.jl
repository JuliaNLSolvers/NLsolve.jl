# MCP example by Josephy.

@testset "mcp_josephy" begin

function f!(F, x)
    F[1]=3*x[1]^2+2*x[1]*x[2]+2*x[2]^2+x[3]+3*x[4]-6
    F[2]=2*x[1]^2+x[1]+x[2]^2+3*x[3]+2*x[4]-2
    F[3]=3*x[1]^2+x[1]*x[2]+2*x[2]^2+2*x[3]+3*x[4]-1
    F[4]=x[1]^2+3*x[2]^2+2*x[3]+3*x[4]-3
end

function j!(J, x)
    J[1,1] = 6*x[1]+2*x[2]
    J[1,2] = 2*x[1]+4*x[2]
    J[1,3] = 1
    J[1,4] = 3
    J[2,1] = 4*x[1]+1
    J[2,2] = 2*x[2]
    J[2,3] = 3
    J[2,4] = 2
    J[3,1] = 6*x[1]+x[2]
    J[3,2] = x[1]+4*x[2]
    J[3,3] = 2
    J[3,4] = 3
    J[4,1] = 2*x[1]
    J[4,2] = 6*x[2]
    J[4,3] = 2
    J[4,4] = 3
end

solution = [ 1.22474487, 0., 0., 0.5 ]

df = OnceDifferentiable(f!, j!, rand(4), rand(4))


# Test smooth reformulation with trust region

r = mcpsolve(df, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :smooth)
@test converged(r)
@test norm(r.zero - solution) < 1e-8

r = mcpsolve(f!, j!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :smooth)
@test converged(r)
@test norm(r.zero - solution) < 1e-8

r = mcpsolve(f!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :smooth)
@test converged(r)
@test norm(r.zero - solution) < 1e-8

r = mcpsolve(f!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :smooth, autodiff = AutoForwardDiff())
@test converged(r)
@test norm(r.zero - solution) < 1e-8


# Test minmax reformulation with trust region

r = mcpsolve(df, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :minmax)
@test converged(r)
@test norm(r.zero - solution) < 1e-8

r = mcpsolve(f!, j!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :minmax)
@test converged(r)
@test norm(r.zero - solution) < 1e-8

r = mcpsolve(f!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :minmax)
@test converged(r)
@test norm(r.zero - solution) < 1e-8

r = mcpsolve(f!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :minmax, autodiff = AutoForwardDiff())
@test converged(r)
@test norm(r.zero - solution) < 1e-8


# Test with Newton method

r = mcpsolve(df, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], method = :newton)
@test converged(r)
@test norm(r.zero - solution) < 1e-8


# An empty box has no solution (#288)

lower = [0.00, 0.00, 0.00, 0.00]
upper = [-1.00, Inf, Inf, Inf]
x0 = [1.25, 0.00, 0.00, 0.50]
for reformulation in (:smooth, :minmax)
    @test_throws "lower[1] = 0.0 is greater than upper[1] = -1.0" mcpsolve(df, lower, upper, x0; reformulation)
    @test_throws ArgumentError mcpsolve(f!, j!, lower, upper, x0; reformulation)
    @test_throws ArgumentError mcpsolve(f!, lower, upper, x0; reformulation)
end

# Equal bounds fix a variable

r = mcpsolve(df, [0.00, 0.00, 0.00, 0.00], [1e20, 0.00, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50])
@test converged(r)
@test norm(r.zero - solution) < 1e-8

# Out-of-place functions

function f_oop(x)
    F = similar(x)
    f!(F, x)
    F
end
function j_oop(x)
    J = zeros(eltype(x), 4, 4)
    j!(J, x)
    J
end
for reformulation in (:smooth, :minmax)
    r = mcpsolve(f_oop, j_oop, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50]; reformulation, inplace = false)
    @test converged(r)
    @test norm(r.zero - solution) < 1e-8

    r = mcpsolve(f_oop, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50]; reformulation, inplace = false)
    @test converged(r)
    @test norm(r.zero - solution) < 1e-8
end

# A fixed variable at a kink of the smooth reformulation (#307). Fixing x[4] at
# 0.5, where F[4] = 0 at the solution, gave a NaN Jacobian.
for method in (:trust_region, :newton)
    r = mcpsolve(f!, [0.00, 0.00, 0.00, 0.50], [Inf, Inf, Inf, 0.50], [1.25, 0.00, 0.00, 0.50]; method)
    @test converged(r)
    @test norm(r.zero - solution) < 1e-8
end

# x[1] is fixed at 1, where F[1] = x[1] - 2 < 0
function f_fixed!(F, x)
    F[1] = x[1] - 2
    F[2] = x[2] - 3
end
function j_fixed!(J, x)
    J .= 0
    J[1, 1] = 1
    J[2, 2] = 1
end
for method in (:trust_region, :newton)
    r = mcpsolve(f_fixed!, j_fixed!, [1.0, -Inf], [1.0, Inf], [1.0, 0.0]; method)
    @test converged(r)
    @test r.zero ≈ [1.0, 3.0]
end

end
