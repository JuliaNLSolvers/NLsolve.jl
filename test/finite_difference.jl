# Test the finite differencing technique

using NLsolve

function f!(x, fvec)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

r = nlsolve(f!, [ 0.1; 1.2], method = :trust_region, iterations = 10, extended_trace = true, show_trace = true)

@assert norm(r.zero - [ 0; 1]) < 1e-8

println(r)
