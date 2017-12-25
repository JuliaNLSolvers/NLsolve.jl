# From Nocedal & Wright, p. 288-289

# Jacobian is singular at the starting point.
# Used to test the behavior of algorithms in that context.

function f_sinj!(F, x)
    F[1] = x[1]
    F[2] = 10*x[1]/(x[1]+convert(eltype(x), 0.1))+2*x[2]^2
end

function g_sinj!(J, x)
    J[1, 1] = 1
    J[1, 2] = 0
    J[2, 1] = 1/(x[1]+convert(eltype(x), 0.1))^2
    J[2, 2] = 4*x[2]
end

df = OnceDifferentiable(f_sinj!, g_sinj!, [3.0, 0.0], [3.0, 0.0])
df32 = OnceDifferentiable(f_sinj!, g_sinj!, [3.0f0, 0.0f0], [3.0f0, 0.0f0])

# Test disabled, not stable across runs
#r = nlsolve(df, [ 3.0; 0], method = :newton, ftol = 1e-5)
#@assert converged(r)
#@assert norm(r.zero) < 1e-5

r = nlsolve(df, [ 3.0; 0.0], method = :trust_region)
@assert converged(r)
@assert norm(r.zero) < 1e-6

r = nlsolve(df32, [3.0f0; 0.0f0], method = :trust_region)
@assert converged(r)
@assert norm(r.zero) < 1e-6

let a = rand(10)
    const A = a*a'
    global f_let!, g_let!
    function f_let!(fvec, x)
        copy!(fvec, A*x)
    end

    function g_let!(fjac, x)
        copy!(fjac, A)
    end
end

df = OnceDifferentiable(f_let!, g_let!, rand(10), rand(10))
r = nlsolve(df, rand(10), method = :trust_region)
