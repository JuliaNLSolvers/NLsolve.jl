using NLsolve

function f(x)
    fvec = zeros(2)
    fjac = zeros(2,2)

    fvec[1] = 3*x[1] + 4*x[2] - 2
    fvec[2] = 5*x[1] - 2*x[2] + 7

    fjac[1, 1] = 3
    fjac[1, 2] = 4
    fjac[2, 1] = 5
    fjac[2, 2] = -2

    return (fvec, fjac)
end

z = newton(f, [ 0.0; 0.0])

@assert norm(z + [ 3.0 4.0; 5.0 -2.0] \ [-2.0; 7]) < 1e-3


           
