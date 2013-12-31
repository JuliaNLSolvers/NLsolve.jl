# Test the finite differencing technique

function f!(x, fvec)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

r = nlsolve(f!, [ 0.1; 1.2])

@assert norm(r.zero - [ 0; 1]) < 1e-8

