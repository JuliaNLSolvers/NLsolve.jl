@testset "fixed points" begin

    # Out of place, no Jacobian, Vector{Float64}. 
    A = [0.7 0.0; 0.0 0.3];
    b = [1.5; 3.2];
    f(x) = A * x + b;
    @test fixedpoint(f, [3.4, 4.3]; inplace = false).zero ≈ [5.0, 4.571428571428571];
    # In place, no Jacobian, Vector{Float64}. 
    function f!(out, x)
        out .= f(x)
    end
    @test fixedpoint(f!, [3.4, 4.3]).zero ≈ [5.0, 4.571428571428571];
    #=
    Needed:
        - StaticArray tests
        - Tests for different container types (Int, etc.)
        - Tests for type stability 
        - Tests for graceful error handling (mismatching dims, types, args, etc.)
        - Tests that the autodifferentiation is working as expected. 
    =#
    # Benchmark simple iteration *******
    # New functions. 
    # In place iterator from the tests. 
    function iterate!(f!, x0; residualnorm = (x -> norm(x,Inf)), tol = 1E-10, maxiter=100)
        residual = Inf
        iter = 1
        xold = x0
        xnew = copy(x0)
        while residual > tol && iter < maxiter
            f!(xnew, xold)
            residual = residualnorm(xold - xnew);
            xold = copy(xnew)
            iter += 1
        end
        return (xold,iter)
    end
    # Out of place iterator from the tests. 
    function iterate(f, x0; residualnorm = (x -> norm(x,Inf)), tol = 1E-10, maxiter=100)
        residual = Inf
        iter = 1
        xold = x0
        xnew = copy(x0)
        while residual > tol && iter < maxiter
            xnew = f(xold)
            residual = residualnorm(xold - xnew);
            xold = copy(xnew)
            iter += 1
        end
        return (xold,iter)
    end
    @test iterate!(f!, [3.4, 4.3])[1] == iterate(f, [3.4, 4.3])[1] ≈ [5.0, 4.571428571428571]
end 