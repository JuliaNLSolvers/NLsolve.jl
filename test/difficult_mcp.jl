# A difficult MCP.
#
# Presented in Miranda and Fackler (2002): "Applied Computational Economics and
# Finance", p. 51

function f!(x, fvec)
    fvec[1] = (1-x[1])^2-1.01
end

function g!(x, fjac)
    fjac[1] = 2(x[1]-1)
end

df = DifferentiableMultivariateFunction(f!, g!)

solution = [ 2.004987562 ]

r = mcpsolve(df, [0.], [Inf], [0.1], method = :newton)
@assert converged(r)
@assert norm(r.zero - solution) < 1e-8
