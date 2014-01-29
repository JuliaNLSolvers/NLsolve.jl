# Verify that we handle correctly the case where the starting point is a zero

function f!(x, fvec)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

df = DifferentiableMultivariateFunction(f!)

r = nlsolve(df, [ 0.; 1.], method = :trust_region)
@assert converged(r)
@assert norm(r.zero - [ 0; 1]) < 1e-8
@assert r.iterations == 0

r = nlsolve(df, [ 0.; 1.], method = :newton)
@assert converged(r)
@assert norm(r.zero - [ 0; 1]) < 1e-8
@assert r.iterations == 0
