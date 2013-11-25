# From Nocedal & Wright, p. 288-289

# Jacobian is singular at the starting point

using NLsolve

function f!(x, fvec)
    fvec[1] = x[1]
    fvec[2] = 10*x[1]/(x[1]+0.1)+2*x[2]^2
end

function g!(x, fjac)
    fjac[1, 1] = 1
    fjac[1, 2] = 0
    fjac[2, 1] = 1/(x[1]+0.1)^2
    fjac[2, 2] = 4*x[2]
end

df = DifferentiableMultivariateFunction(f!, g!)

r = nlsolve(df, [ 3.0; 0], show_trace = true)

@assert norm(r.zero) < 1e-6

println(r)
