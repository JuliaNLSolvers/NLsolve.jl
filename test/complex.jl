
@testset "complex" begin
function f!(F, x)
  F[1] = x[1]*x[2] + (1+im)
  F[2] = x[1]^2 + x[2]^2 - (2-3im)
end
function f_real!(F::AbstractArray{T}, x::AbstractArray{T}) where {T<:Real}
  f!(reinterpret(Complex{T}, F), reinterpret(Complex{T}, x))
end

for alg in [:newton,:trust_region,:anderson] # TODO add broyden
    sol = nlsolve(f!, [1.0+0.1im, 2+1im], method = alg, store_trace=true, extended_trace=true, iterations=100, m=10, beta=0.01)
    sol_real = nlsolve(f_real!, reinterpret(Float64, [1.0+0.1im, 2+1im]), method = alg, store_trace=true, extended_trace=true, iterations=100, m=10, beta=0.01)

    @test converged(sol) == converged(sol_real)
    @test sol.zero ≈ reinterpret(ComplexF64, sol_real.zero)
    if alg in (:newton, :trust_region) #those are supposed to be exactly the same (in exact arithmetic)
        @test sol.iterations == sol_real.iterations
        @test sol.f_calls == sol_real.f_calls
        @test sol.g_calls == sol_real.g_calls
        @test all(sol_real.trace[i].stepnorm == sol_real.trace[i].stepnorm for i in 2:sol.iterations)
        @test all(norm(sol.trace[i].metadata["f(x)"]) ≈ norm(sol_real.trace[i].metadata["f(x)"]) for i in 1:5)
    end
end
for linesearches in [BackTracking(),StrongWolfe(),HagerZhang(),MoreThuente()] #Static is already included in default Newton without linesearch
    sol = nlsolve(f!, [1.0+0.1im, 2+1im], method = :newton, linesearch=linesearches,store_trace=true, extended_trace=true, iterations=100, m=10, beta=0.01)
    sol_real = nlsolve(f_real!, reinterpret(Float64, [1.0+0.1im, 2+1im]), method = :newton, linesearch=linesearches, store_trace=true, extended_trace=true, iterations=100, m=10, beta=0.01)
    @test converged(sol) == converged(sol_real)
    @test sol.zero ≈ reinterpret(ComplexF64, sol_real.zero)
    @test sol.iterations == sol_real.iterations
    @test sol.f_calls == sol_real.f_calls
    @test sol.g_calls == sol_real.g_calls
    @test all(sol_real.trace[i].stepnorm == sol_real.trace[i].stepnorm for i in 2:sol.iterations)
    @test all(norm(sol.trace[i].metadata["f(x)"]) ≈ norm(sol_real.trace[i].metadata["f(x)"]) for i in 1:5)
    NLsolve.x_trace(sol_real)
    NLsolve.F_trace(sol_real)
    NLsolve.J_trace(sol_real)
end
end
