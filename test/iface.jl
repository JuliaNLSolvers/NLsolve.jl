# Test all the various ways of specifying a function

function f!(x, fvec)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

function g!(x, fjac)
    fjac[1, 1] = x[2]^3-7
    fjac[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    fjac[2, 1] = x[2]*u
    fjac[2, 2] = u
end

function fg!(x, fvec, fjac)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
    fjac[1, 1] = x[2]^3-7
    fjac[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    fjac[2, 1] = x[2]*u
    fjac[2, 2] = u
end

r = nlsolve(DifferentiableMultivariateFunction(f!), [ -0.5; 1.4])
@assert converged(r)

r = nlsolve(DifferentiableMultivariateFunction(f!, g!), [ -0.5; 1.4])
@assert converged(r)

r = nlsolve(DifferentiableMultivariateFunction(f!, g!, fg!), [ -0.5; 1.4])
@assert converged(r)

r = nlsolve(f!, [ -0.5; 1.4])
@assert converged(r)

r = nlsolve(f!, g!, [ -0.5; 1.4])
@assert converged(r)
