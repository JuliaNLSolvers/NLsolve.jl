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

r = nlsolve(OnceDifferentiable(f!, [ -0.5; 1.4], [ -0.5; 1.4]), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(OnceDifferentiable(f!, j!, [ -0.5; 1.4], [ -0.5; 1.4]), [ -0.5; 1.4])
@test converged(r)

r = nlsolve(OnceDifferentiable(f!, j!, fj!, [ -0.5; 1.4], [ -0.5; 1.4]),  [ -0.5; 1.4])
@test converged(r)

r = nlsolve(f!, [ -0.5; 1.4])
@test converged(r)

r = nlsolve(f!, j!, [ -0.5; 1.4])
@test converged(r)

#r = nlsolve(only_f!_and_fj!(f!, fj!), [ -0.5; 1.4])
#@test converged(r)

#r = nlsolve(only_fj!(fj!), [ -0.5; 1.4])
#@test converged(r)


# Use not-in-place forms
function f(x)
    F = Array{eltype(x)}(undef, 2)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
    return F
end

function g(x)
    J = Array{eltype(x)}(undef, 2, 2)
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


r = nlsolve(f, [ -0.5; 1.4]; inplace = false)
@test converged(r)

r = nlsolve(f, [ -0.5; 1.4]; inplace = false, autodiff = true)
@test converged(r)

r = nlsolve(f, g, [ -0.5; 1.4]; inplace = false)
@test converged(r)

r = nlsolve(f, g, fg, [ -0.5; 1.4]; inplace = false)
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

@testset "andersont trace issue #160" begin

    function f_2by2!(F, x)
        F[1] = (x[1]+3)*(x[2]^3-7)+18
        F[2] = sin(x[2]*exp(x[1])-1)
    end

    function g_2by2!(J, x)
        J[1, 1] = x[2]^3-7
        J[1, 2] = 3*x[2]^2*(x[1]+3)
        u = exp(x[1])*cos(x[2]*exp(x[1])-1)
        J[2, 1] = x[2]*u
        J[2, 2] = u
    end

    df = OnceDifferentiable(f_2by2!, g_2by2!, [ -0.5; 1.4], [ -0.5; 1.4])

    r = nlsolve(df, [ 0.01; .99], method = :anderson, m = 10, beta=.01, show_trace=true)
end
end
