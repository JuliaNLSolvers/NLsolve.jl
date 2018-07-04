@testset "fixed points" begin

    @testset "anderson/simple iteration" begin 
        # Out of place, no Jacobian, Vector{Float64}. 
        A = [0.7 0.0; 0.0 0.3];
        b = [1.5; 3.2];
        f(x) = A * x + b;
        @test fixedpoint(f, [3.4, 4.3]; inplace = false).zero ≈ [5.0, 4.571428571428571];
        # In place, no Jacobian, Vector{Float64}. 
        function f!(out, x)
            out .= f(x)
        end
        @test fixedpoint(f!, [3.4, 4.3]).zero ≈ [5.0, 4.571428571428571];; 
        #=
        Needed:
            - StaticArray tests
            - Tests for different container types (Int, etc.)
            - Tests for type stability 
            - Tests for graceful error handling (mismatching dims, types, args, etc.)
            - Tests that the autodifferentiation is working as expected. 
        =#
        #= 
        Notes for Ali: 
            - One way to do the tests for a bunch of different types is to write something like Array{T}, where T is filled by looping over (say) Float64, Int64, Vectors, ... Because types in Julia work as objects, this is kosher. 
            - Static Arrays come from: https://github.com/JuliaArrays/StaticArrays.jl. We should grab the Julia 0.6 one. Because these aren't allowed to change, we should do something nice if someone tries to give us a in place function on a static array. 
            - If you want to test where something should fail, you can use the macro "@test_throws." So, for example, if there's blahblahblah throws a MethodError, we can say:
            @test_throws MethodError blahblahblah
            - If you want to see the output for something function in the terminal or REPL, you can put the @show macro inside the function. For example: @show arg would display arg. Otherwise the output for passing tests is suppressed. 
        =# 
    end 
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