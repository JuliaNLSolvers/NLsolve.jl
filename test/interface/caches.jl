@testset "caches" begin

function f!(fvec, x)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

df = OnceDifferentiable(f!, rand(2), rand(2))

# Test that the cache is actually passed all the way by verifying that it's
# modified during the solve calls. 
nc = NLsolve.NewtonCache(df)
nc.p .= 22.0
ncp = copy(nc.p)
r = NLsolve.newton(df, [ 1.; 1.], 0.1, 0.1, 100, false, false, false, LineSearches.Static(), nc)
@test !(ncp == nc.p)

ac = NLsolve.AndersonCache(df, NLsolve.Anderson(10, 0.9))
ac.xs .= 22.0
acxs = copy(ac.xs)
r = NLsolve.anderson(df, [ 1.; 1.], 0.1, 0.1, 100, false, false, false, 10, 0.9, ac)
@test !(acxs == ac.xs)

ntc = NLsolve.NewtonTrustRegionCache(df)
ntc.r_predict .= 22.0
ntcr_predict = copy(ntc.r_predict)
r = NLsolve.trust_region(df, [ 1.; 1.], 0.1, 0.1, 100, false, false, false, 1.0, true, ntc)
@test !(ntcr_predict == ntc.r_predict)

end
