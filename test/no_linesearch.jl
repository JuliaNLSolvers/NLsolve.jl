@testset "no linesearch" begin

function f_nolin!(fvec, x)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

r = nlsolve(f_nolin!, [ -0.5; 1.4], autodiff = true, method = :newton, linesearch! = NLsolve.no_linesearch!)
@test converged(r)

end
