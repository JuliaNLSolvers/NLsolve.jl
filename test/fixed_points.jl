# Test fixed points
@testset "fixed_points" begin
using NLsolve, Base.Test, StaticArrays
#TASK 1: Anderson

#matrix valued function
A = [0.5 0; 0 0.2]
b = [1.0; 2.0]
f(x) = A * x + b
fixedpoint(f,[1.0; 0.1], :anderson)

#inplace matrix valued function
A = [0.5 0; 0 0.2]
b = [1.0; 2.0]
function f!(x, fx)
        fx[:] = A * x + b
end
fixedpoint(f!,[1.0; 0.2],:anderson)


# Variation using static arrays
A= SMatrix{2,2}(0.5,0,0,0.2)
b = SVector(1.0, 2.0)
f(x)= A * x + b
fixedpoint(g,[1.0; 0.2],:anderson) # how to call new interface





#TASK 2: Simple Iteration

#The inplace one is not implemented, so left in the old code.
function fixedpointOLD!(f!, x0; residualnorm = (x -> norm(x,Inf)), tol = 1E-10, maxiter=100)
    residual = Inf
    iter = 1
    xold = x0
    xnew = copy(x0)
    while residual > tol && iter < maxiter
        f!(xold, xnew)
        residual = residualnorm(xold - xnew);
        xold = copy(xnew)
        iter += 1
    end
    return (xold,iter)
end


# A matrix equation fixedpoint(f,[1.0; 0.1])
A = [0.5 0; 0 0.2]
b = [1.0; 2.0]
f(x) = A * x + b
@test fixedpoint(f, method =:simple_iteration,[1.0; 0.1], iter=100,store_trace==true,show_trace==true,
                 extended_trace=true)[1] ≈ [2.0;2.5]


r = fixedpoint(f,[1.0; 0.1], method = :simple_iteration)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8


#@test fixedpoint(f,[1.0; 0.1]; iterations=100)[1] ≈ [2.0;2.5]

@test fixedpoint(f,[1.0; 0.1]; iterations=100)[1] ≈ [2.0;2.5]
A = [0.5 0; 0 0.2]
b = [1.0; 2.0]
function f!(x, fx)
        fx[:] = A * x + b
end
#fixedpointOLD!(f!,[1.0; 0.2])

A= SMatrix{2,2}(0.5,0,0,0.2)
b = SVector(1.0, 2.0)
f(x)= A * x + b
@test fixedpoint(f,[1.0; 0.2],method =:simple_iteration,[1.0; 0.1],store_trace==true,show_trace==true,
                 extended_trace=true)[1] ≈ [2.0;2.5]
