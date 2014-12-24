# MCP example by Josephy.

function f!(x, fvec)
    fvec[1]=3*x[1]^2+2*x[1]*x[2]+2*x[2]^2+x[3]+3*x[4]-6
    fvec[2]=2*x[1]^2+x[1]+x[2]^2+3*x[3]+2*x[4]-2
    fvec[3]=3*x[1]^2+x[1]*x[2]+2*x[2]^2+2*x[3]+3*x[4]-1
    fvec[4]=x[1]^2+3*x[2]^2+2*x[3]+3*x[4]-3
end

function g!(x, fjac)
    fjac[1,1] = 6*x[1]+2*x[2]
    fjac[1,2] = 2*x[1]+4*x[2]
    fjac[1,3] = 1
    fjac[1,4] = 3
    fjac[2,1] = 4*x[1]+1
    fjac[2,2] = 2*x[2]
    fjac[2,3] = 3
    fjac[2,4] = 2
    fjac[3,1] = 6*x[1]+x[2]
    fjac[3,2] = x[1]+4*x[2]
    fjac[3,3] = 2
    fjac[3,4] = 3
    fjac[4,1] = 2*x[1]
    fjac[4,2] = 6*x[2]
    fjac[4,3] = 2
    fjac[4,4] = 3
end

solution = [ 1.22474487, 0., 0., 0.5 ]

df = DifferentiableMultivariateFunction(f!, g!)

r = mcpsolve(df, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50])
@assert converged(r)
@assert norm(r.zero - solution) < 1e-8

r = mcpsolve(df, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50], method = :newton)
@assert converged(r)
@assert norm(r.zero - solution) < 1e-8

r = mcpsolve(f!, [0.00, 0.00, 0.00, 0.00], [1e20, 1e20, 1e20, 1e20], [1.25, 0.00, 0.00, 0.50])
@assert converged(r)
@assert norm(r.zero - solution) < 1e-8

