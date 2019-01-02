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
    res_default_solve = nlsolve(df, prob[2]; method=:newton)
    prob = rosenbrock()
    df = prob[1]
    res_gmres_solve = nlsolve(df, prob[2]; method=:newton, linsolve = (x, A, b)->gmres!(fill!(x, eltype(x)(0)), A, b))
    @test res_gmres_solve.zero ≈ res_default_solve.zero
    prob = rosenbrock()
    df = prob[1]
    res_idrs_solve = nlsolve(df, prob[2]; method=:newton, linsolve = (x, A, b)->idrs!(fill!(x, eltype(x)(0)), A, b))
    @test res_idrs_solve.zero ≈ res_default_solve.zero
    testy = [false]
    function loud_solve(x, A, b)
        testy[1] = true
        gmres!(x, A, b)
    end
    prob = rosenbrock()
    df = prob[1]
    nlsolve(df, prob[2]; method=:newton, linsolve = loud_solve)
    @test testy[1] == true
end
