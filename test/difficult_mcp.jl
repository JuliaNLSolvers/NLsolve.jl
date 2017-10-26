# A difficult MCP.
#
# Presented in Miranda and Fackler (2002): "Applied Computational Economics and
# Finance", p. 51
@testset "difficult_mcp" begin
function f_diffmcp!(fvec, x)
    fvec[1] = (1-x[1])^2-1.01
end

function g_diffmcp!(fjac, x)
    fjac[1] = 2(x[1]-1)
end

df = DifferentiableVector(f_diffmcp!, g_diffmcp!)

solution = [ 2.004987562 ]

# TODO Figure out a good linesearch to make this work again
#r = mcpsolve(df, [0.], [Inf], [0.1], method = :newton)
#@test converged(r)
#@test norm(r.zero - solution) < 1e-8
end
