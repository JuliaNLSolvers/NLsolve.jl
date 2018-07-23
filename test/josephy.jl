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

r = mcpsolve(f!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :smooth, autodiff = :forward)
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

r = mcpsolve(f!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], reformulation = :minmax, autodiff = :forward)
@test converged(r)
@test norm(r.zero - solution) < 1e-8


# Test with Newton method

r = mcpsolve(df, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], method = :newton)
@test converged(r)
@test norm(r.zero - solution) < 1e-8

end
