@testset "fixed points" begin

    # Basic container tests *******
    for T in (Float64, Int64, BitFloat, BigInt)
        # Write an @test for out of place vector 
        # Write an @test for in place vector 
        # Write an @test for out of place matrix 
        # Write an @test for in place matrix 
        # Write an @test for out of place n_ary scalar
        # Write an @test for in place n_ary scalar 
    end 

    # StaticArray tests. ************
    for T in (Float64, Int64, BitFloat, BigInt)
    end 

    # Error tests **********
    # @test_throws for DimensionMismatch on the f(x) - x 
    # @test for DimensionMismatch on the out .-= x 
    
    # Type inference tests ************
    # Can just do @code_warntype for different function calls here 

    # Precision tests ************
    # Test some of the floating point math (if necessary).

    # Benchmarking **********
    # Should do one for different kinds of types, in place vs. out of place, big vs small matrices, etc. 
    
    # M Values tests **********
    # Should probably play around with different m values and see that nothing dramatically breaks. 

    # Actual tests for now ***********
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


    # Simple iteration stuff ****************
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