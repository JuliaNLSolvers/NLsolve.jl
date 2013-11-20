using NLsolve

function f!(x, fvec)
    fvec[1] = 3*x[1] + 4*x[2] - 2
    fvec[2] = 5*x[1] - 2*x[2] + 7
end

function g!(x, fjac)
    fjac[1, 1] = 3
    fjac[1, 2] = 4
    fjac[2, 1] = 5
    fjac[2, 2] = -2
end

df = DifferentiableMultivariateFunction(f!, g!)

r = nlsolve(df, [ 0.0; 0.0], show_trace = true)

@assert norm(r.zero + [ 3.0 4.0; 5.0 -2.0] \ [-2.0; 7]) < 1e-3

println(r)


           
