# From Nocedal & Wright, p. 281

using NLsolve

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

df = DifferentiableMultivariateFunction(f!, g!)

r = nlsolve(df, [ -0.5; 1.4], show_trace = true)

@assert norm(r.zero - [ 0; 1]) < 1e-8

println(r)
