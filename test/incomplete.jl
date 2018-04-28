@testset "incomplete objective" begin
    function myfun!(F, J, x)
           if !(F == nothing)
               F[1] = (x[1]+3)*(x[2]^3-7)+18
               F[2] = sin(x[2]*exp(x[1])-1)
           end

           if !(J == nothing)
               J[1, 1] = x[2]^3-7
               J[1, 2] = 3*x[2]^2*(x[1]+3)
               u = exp(x[1])*cos(x[2]*exp(x[1])-1)
               J[2, 1] = x[2]*u
               J[2, 2] = u
           end
    end

    r = nlsolve(only_fj!(myfun!), [0.5, 0.5])
    @test norm(r.zero - [0.0, 1.0], Inf) < 1e-8

    function myfun(x)
        F = similar(x)
        F[1] = (x[1]+3)*(x[2]^3-7)+18
        F[2] = sin(x[2]*exp(x[1])-1)


        J = NLSolversBase.alloc_DF(x, F)
        J[1, 1] = x[2]^3-7
        J[1, 2] = 3*x[2]^2*(x[1]+3)
        u = exp(x[1])*cos(x[2]*exp(x[1])-1)
        J[2, 1] = x[2]*u
        J[2, 2] = u

        F, J
    end
    r = nlsolve(only_fj(myfun), [0.5, 0.5])
    @test norm(r.zero - [0.0, 1.0], Inf) < 1e-8
end
