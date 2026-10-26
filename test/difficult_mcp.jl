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

df = OnceDifferentiable(f_diffmcp!, g_diffmcp!, [0.0], [0.0])

solution = [ 2.004987562 ]

# From x0 = 0.1, Newton with the default BackTracking line search stalls near
# x = -0.005, outside the box, where the reformulated residual is about 0.01.
# The step shrinks to exactly zero, which passes the step test when xtol = 0, so
# the run reports convergence at a point that is not a solution (#319).
r = mcpsolve(df, [0.], [Inf], [0.1], method = :newton)
@test_broken !converged(r)
@test_broken norm(r.zero - solution) < 1e-8

# From x0 = 1.5 both methods and both reformulations reach the solution
for method in (:newton, :trust_region), reformulation in (:smooth, :minmax)
    r = mcpsolve(df, [0.], [Inf], [1.5]; method, reformulation)
    @test converged(r)
    @test norm(r.zero - solution) < 1e-8
end
end
