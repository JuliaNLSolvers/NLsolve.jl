@testset "complex" begin
function f!(F, x)
  F[1] = x[1]*x[2] + 1
  F[2] = x[1]^2 + x[2]^2 - 2
end
function f_real!(F::AbstractArray{T}, x::AbstractArray{T}) where {T<:Real}
  f!(reinterpret(Complex{T}, F), reinterpret(Complex{T}, x))
end

for alg in [:trust_region, :newton]
    sol = nlsolve(f!, [1.0+0.1im, 2+1im], method = alg, store_trace=true, extended_trace=true)
    sol_real = nlsolve(f_real!, reinterpret(Float64, [1.0+0.1im, 2+1im]), method = alg, store_trace=true, extended_trace=true)

    @test converged(sol) == converged(sol_real)
    @test sol.zero ≈ reinterpret(ComplexF64, sol_real.zero)
    @test sol.iterations == sol_real.iterations
    @test sol.f_calls == sol_real.f_calls
    @test sol.g_calls == sol_real.g_calls
    @test all(sol_real.trace[i].stepnorm == sol_real.trace[i].stepnorm for i in 2:sol.iterations)
    @test all(norm(sol.trace[i].metadata["f(x)"]) ≈ norm(sol_real.trace[i].metadata["f(x)"]) for i in 1:5)
end
end
