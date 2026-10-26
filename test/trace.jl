# The trace reports the 2-norm of the step, not its square

@testset "trace step norm" begin

function f_2by2!(F, x)
    F[1] = (x[1] + 3) * (x[2]^3 - 7) + 18
    F[2] = sin(x[2] * exp(x[1]) - 1)
end
x0 = [-0.5, 1.4]

for method in (:newton, :broyden, :trust_region)
    r = nlsolve(f_2by2!, x0; method, iterations = 20, store_trace = true, extended_trace = true)
    states = r.trace.states
    @test length(states) > 2
    for k in 2:length(states)
        @test states[k].stepnorm ≈ norm(states[k].metadata["x"] - states[k-1].metadata["x"])
    end
end

# Anderson's step is g(x) - x = beta * f(x)
f_lin!(F, x) = (F .= [1.0, 2.0] .- x)
beta = 0.5
r = nlsolve(f_lin!, [0.0, 0.0]; method = :anderson, beta, iterations = 10, store_trace = true, extended_trace = true)
states = filter(s -> !isnan(s.stepnorm), r.trace.states)
@test length(states) > 1
for s in states
    @test s.stepnorm ≈ beta * norm(s.metadata["f(x)"])
end

end
