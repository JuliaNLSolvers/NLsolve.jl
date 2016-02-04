# Test automatic differentiation
@testset "autodiff" begin
function f!(x, fvec)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

r = nlsolve(f!, [ -0.5; 1.4], autodiff = true)
@test converged(r)
@test norm(r.zero - [ 0; 1]) < 1e-8

@test NLsolve.compute_chunk_size(12) == 6
@test NLsolve.compute_chunk_size(13) == 1
@test NLsolve.compute_chunk_size(9) == 9
@test NLsolve.compute_chunk_size(25) == 5
@test NLsolve.compute_chunk_size(20) == 10
@test NLsolve.compute_chunk_size(1) == 1

end
