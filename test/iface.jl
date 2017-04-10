# Test all the various ways of specifying a function

@testset "iface" begin

# Using functions modifying in-place

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
@test converged(r)

r = nlsolve(DifferentiableMultivariateFunction(f!, g!), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(DifferentiableMultivariateFunction(f!, g!, fg!), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(f!, [ -0.5; 1.4])
@test converged(r)

r = nlsolve(f!, g!, [ -0.5; 1.4])
@test converged(r)

r = nlsolve(only_f!_and_fg!(f!, fg!), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(only_fg!(fg!), [ -0.5; 1.4])
@test converged(r)


# Using functions returning their output

function f(x)
    fvec = Array(eltype(x), 2)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
    return(fvec)
end

function g(x)
    fjac = Array(eltype(x), 2, 2)
    fjac[1, 1] = x[2]^3-7
    fjac[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    fjac[2, 1] = x[2]*u
    fjac[2, 2] = u
    return(fjac)
end

function fg(x)
    fvec = Array(eltype(x), 2)
    fjac = Array(eltype(x), 2, 2)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
    fjac[1, 1] = x[2]^3-7
    fjac[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    fjac[2, 1] = x[2]*u
    fjac[2, 2] = u
    return(fvec,fjac)
end

r = nlsolve(not_in_place(f), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(not_in_place(f), [ -0.5; 1.4], autodiff = true)
@test converged(r)

r = nlsolve(not_in_place(f), not_in_place(g), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(not_in_place(f, g, fg), [ -0.5; 1.4])
@test converged(r)


# Using functions taking scalar as inputs

function f(x, y)
    fx = (x+3)*(y^3-7)+18
    fy = sin(y*exp(x)-1)
    return(fx,fy)
end

r = nlsolve(n_ary(f), [ -0.5; 1.4])
@test converged(r)
r = nlsolve(n_ary(f), [ -0.5; 1.4], autodiff = true)
@test converged(r)

end