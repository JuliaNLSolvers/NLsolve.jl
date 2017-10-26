# This file tests if the number of f and g calls are correctly counted
@testset "f_g_counts" begin

    fcalls = Ref(0)
    gcalls = Ref(0)

    function f_counts_ref!(x::Vector, fvec::Vector, fcalls)
        fcalls[] += 1
        fvec[1] = 1 - x[1]
        fvec[2] = 10(x[2]-x[1]^2)
    end
    f_counts!(fvec, x) = f_counts_ref!(x, fvec, fcalls)

    function g_counts_ref!(x::Vector, fjac::Matrix, gcalls)
        gcalls[] += 1
        fjac[1,1] = -1
        fjac[1,2] = 0
        fjac[2,1] = -20x[1]
        fjac[2,2] = 10
    end
    g_counts!(fjac, x) = g_counts_ref!(x, fjac, gcalls)

    df = DifferentiableVector(f_counts!, g_counts!)

    x0 = [-1.2; 1.]

    fcalls[] = 0
    gcalls[] = 0

    r = nlsolve(df, x0, method = :trust_region)
    @test r.f_calls == fcalls[]
    @test r.g_calls == gcalls[]
    df = DifferentiableVector(f_counts!, g_counts!)

    fcalls[] = 0
    gcalls[] = 0

    r = nlsolve(df, x0, method = :newton)
    @test r.f_calls == fcalls[]
    @test r.g_calls == gcalls[]
end
