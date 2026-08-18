@testset "complex" begin
function f!(F, x)
  F[1] = x[1]*x[2] + (1+im)
  F[2] = x[1]^2 + x[2]^2 - (2-3im)
end
function f_real!(F::AbstractArray{T}, x::AbstractArray{T}) where {T<:Real}
  f!(reinterpret(Complex{T}, F), reinterpret(Complex{T}, x))
end

solver_kwargs = (store_trace = true, extended_trace = true, iterations = 100, m = 10, beta = 0.01)

function agrees_with_real_embedding(sol, method, linesearch)
    sol_real = nlsolve(f_real!, reinterpret(Float64, [1.0+0.1im, 2+1im]);
                       method = method, linesearch = linesearch, solver_kwargs...)
    agrees = converged(sol) == converged(sol_real) &&
             sol.zero ≈ reinterpret(ComplexF64, sol_real.zero)
    if method in (:newton, :trust_region) # these are supposed to be exactly the same (in exact arithmetic)
        agrees &= sol.iterations == sol_real.iterations &&
                  sol.f_calls == sol_real.f_calls &&
                  sol.g_calls == sol_real.g_calls &&
                  all(sol_real.trace[i].stepnorm == sol_real.trace[i].stepnorm for i in 2:sol.iterations) &&
                  all(norm(sol.trace[i].metadata["f(x)"]) ≈ norm(sol_real.trace[i].metadata["f(x)"]) for i in 1:5)
    end
    return agrees
end

for method in [:newton, :trust_region, :anderson] # TODO add broyden
    sol = nlsolve(f!, [1.0+0.1im, 2+1im]; method = method, solver_kwargs...)
    @test converged(sol)
    @test sol.residual_norm < 1e-8
    @test agrees_with_real_embedding(sol, method, Static())
end

for linesearch in [BackTracking(), StrongWolfe(), HagerZhang(), MoreThuente()] # Static is covered above
    sol = nlsolve(f!, [1.0+0.1im, 2+1im]; method = :newton, linesearch = linesearch, solver_kwargs...)
    @test converged(sol)
    @test sol.residual_norm < 1e-8
    @test agrees_with_real_embedding(sol, :newton, linesearch)
end
end
