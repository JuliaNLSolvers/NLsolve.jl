@testset "singular case" begin
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
r = nlsolve(df, [ 3.0; 0], method = :newton, ftol = 1e-5)
#@assert converged(r)
#@assert norm(r.zero) < 1e-5

r = nlsolve(df, [ 3.0; 0.0], method = :trust_region)
@test converged(r)
@test norm(r.zero) < 1e-6

r = nlsolve(df32, [3.0f0; 0.0f0], method = :trust_region)
@test converged(r)
@test norm(r.zero) < 1e-6

r = nlsolve(df, [ 3.0; 0.0], method = :broyden)
@test converged(r)
@test_broken norm(r.zero) < 1e-6

r = nlsolve(df32, [3.0f0; 0.0f0], method = :broyden)
@test converged(r)
@test_broken norm(r.zero) < 1e-6

let a = rand(10)
    A = a*a'
    global f_let!, g_let!
    function f_let!(fvec, x)
        copyto!(fvec, A*x)
    end

    function g_let!(fjac, x)
        copyto!(fjac, A)
    end
end

df = OnceDifferentiable(f_let!, g_let!, rand(10), rand(10))
r = nlsolve(df, rand(10), method = :trust_region)
end

@testset "underflowing column norms" begin
# https://github.com/JuliaNLSolvers/NLsolve.jl/issues/297
# A Jacobian column with entries below sqrt(floatmin) is nonzero, so it passed
# the d[j] == 0 guard, but d[j]^2 underflows to zero and g ./ d.^2 in dogleg!
# produced NaN.

function f_tiny!(F, x)
    F[1] = x[1] - 2
    F[2] = 1e-200 * (x[2] - 3)
end

function g_tiny!(J, x)
    J[1, 1] = 1.0
    J[1, 2] = 0.0
    J[2, 1] = 0.0
    J[2, 2] = 1e-200
end

df_tiny = OnceDifferentiable(f_tiny!, g_tiny!, zeros(2), zeros(2))

r = nlsolve(df_tiny, zeros(2), method = :trust_region)
@test converged(r)
@test !any(isnan, r.zero)
@test r.zero ≈ [2.0, 3.0]

r = nlsolve(df_tiny, zeros(2), method = :trust_region, autoscale = false)
@test !any(isnan, r.zero)
end
