@testset "fixed points" begin

#= 
    Global objects.  
=#

# Function converter for out of place --> in place. 
function make_inplace(f::Function)
    function inplace_f(out, x)
        out .= f(x) 
    end
    return inplace_f
end 

# "Naive" simple iteration methods for comparison. 
    # In place. 
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
    # Out of place.
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
    # Tests of these. 
    A_1 = [0.7 0.0; 0.0 0.3];
    b_1 = [1.5; 3.2];
    f_1 = x -> A_1 * x + b_1; 
    f_1! = make_inplace(f_1)
    init_x = [3.4, 4.3]
    @test iterate!(f_1!, init_x)[1] == iterate(f_1, init_x)[1] ≈ [5.0, 4.571428571428571]

#= 
    Anderson tests. 
=#

# Tests against the above example. 
    @test fixedpoint(f_1!, [3.4, 4.3]).zero == fixedpoint(f_1, [3.4, 4.3]; inplace = false).zero ≈ [5.0, 4.571428571428571]
    @test fixedpoint(f_1!, [3.4, 4.3]; m = 2).zero == fixedpoint(f_1, [3.4, 4.3]; inplace = false, m = 2).zero ≈ [5.0, 4.571428571428571]

# Tests for some common functions. 
    # x -> sin.(x) 
    f_2 = x -> 0.5 * x
    f_2! = make_inplace(f_2)
    srand = 123 # For determinism in the random tests. 
    init_x2 = rand(Float64, 4)
    @test fixedpoint(f_2!, init_x2; iterations = 10000, ftol = 1e-15).zero == fixedpoint(f_2, init_x2; inplace = false, iterations = 10000, ftol = 1e-15).zero 
    @test isapprox(fixedpoint(f_2!, init_x2; iterations = 10000, ftol = 1e-15).zero, zeros(Float64, 4), atol = 1e-10)
    # x -> exp(-x)
    f_3 = x -> exp.(-x)
    f_3! = make_inplace(f_3)
    init_x3 = [rand(Float64)*100]
    @test fixedpoint(f_3!, init_x3).zero == fixedpoint(f_3, init_x3; inplace = false).zero ≈ [0.5671432953088511]
    @test fixedpoint(f_3!, init_x3; m = 4).zero == fixedpoint(f_3, init_x3; inplace = false, m = 4).zero ≈ [0.5671432953088511]

# Tests for...

#= 
    Gradient method tests. 
=#
    # Autodifferentiation tests. 
    @test fixedpoint(f_3!, init_x3; autodiff = :forward).zero == fixedpoint(f_3, init_x3; inplace = false, autodiff = :forward).zero ≈ [0.5671432953088511]
    @test fixedpoint(f_3!, init_x3; autodiff = :central).zero == fixedpoint(f_3, init_x3; inplace = false, autodiff = :central).zero ≈ [0.5671432953088511]

    # Newton tests. 
    @test fixedpoint(f_3!, init_x3; autodiff = :forward, method = :newton).zero == fixedpoint(f_3, init_x3; inplace = false, autodiff = :forward, method = :newton).zero ≈ [0.5671432953088511]
    @test fixedpoint(f_3!, init_x3; autodiff = :central, method = :newton).zero == fixedpoint(f_3, init_x3; inplace = false, autodiff = :central, method = :newton).zero ≈ [0.5671432953088511]



end 