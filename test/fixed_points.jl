# Test fixed points 

using NLsolve

function fixedpoint(f, x0; residualnorm = (x -> norm(x,Inf)), tol = 1E-10, maxiter=100)    
    residual = Inf
    iter = 1
    xold = x0
    while residual > tol && iter < maxiter
        xnew = f(xold)        
        residual = residualnorm(xold - xnew);
        xold = xnew
        iter += 1
    end
    return (xold,iter)
end

function fixedpoint!(f!, x0; residualnorm = (x -> norm(x,Inf)), tol = 1E-10, maxiter=100)    
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

fixedpoint(x -> 0.5x + 1, 1.0)
fixedpoint(x -> 1.1x + 1, 1.0)

# test slightly less trival mappings 

A = [0.5 0; 0 0.2]
b = [1.0; 2.0]
f(x) = A * x + b  # A matrix equation
fixedpoint(f,[1.0; 0.1])


A = [0.5 0; 0 0.2]
b = [1.0; 2.0]
function f!(x, fx)
        fx[:] = A * x + b
end
fixedpoint!(f!,[1.0; 0.2])


Pkg.add("StaticArrays")
using StaticArrays
A= SMatrix{2,2}(0.5,0,0,0.2)
b = SVector(1.0, 2.0)
f(x)= A * x + b 
fixedpoint(f,[1.0; 0.1])
