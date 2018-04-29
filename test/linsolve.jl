@testset "linsolve" begin

    function rosenbrock()
        function f!(fvec, x)
            fvec[1] = 1 - x[1]
            fvec[2] = 10(x[2]-x[1]^2)
        end
        function j!(fjac, x)
            fjac[1,1] = -1
            fjac[1,2] = 0
            fjac[2,1] = -20x[1]
            fjac[2,2] = 10
        end
        (OnceDifferentiable(f!, j!, [-1.2, 1.0], [-1.2, 1.0]), [-1.2; 1.0], "Rosenbrock")
    end
    prob = rosenbrock()
    df = prob[1]
    nlsolve(df, prob[2], NLsolve.Newton(linsolve = (x,A,b)->gmres!(x, A,b)))
end
