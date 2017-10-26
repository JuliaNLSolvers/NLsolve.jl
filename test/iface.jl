# Test all the various ways of specifying a function

@testset "iface" begin

# Using functions modifying in-place

function f!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

function j!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end

function fj!(F, J, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end

r = nlsolve(DifferentiableVector(f!), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(DifferentiableVector(f!, j!), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(DifferentiableVector(f!, j!, fj!), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(f!, [ -0.5; 1.4])
@test converged(r)

r = nlsolve(f!, j!, [ -0.5; 1.4])
@test converged(r)

r = nlsolve(only_f!_and_fj!(f!, fj!), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(only_fj!(fj!), [ -0.5; 1.4])
@test converged(r)


# Use not-in-place forms
function f(x)
    F = Array{eltype(x)}(2)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
    return F
end

function g(x)
    J = Array{eltype(x)}(2, 2)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
    return J
end

function fg(x)
    F = Array{eltype(x)}(2)
    J = Array{eltype(x)}(2, 2)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
    return F, J
end

r = nlsolve(not_in_place(f), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(not_in_place(f), [ -0.5; 1.4], autodiff = true)
@test converged(r)

r = nlsolve(not_in_place(f, g), [ -0.5; 1.4])
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
