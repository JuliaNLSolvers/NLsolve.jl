function autodiff(f!, initial_x::AbstractArray,
                  jac_cfg::ForwardDiff.JacobianConfig=ForwardDiff.JacobianConfig(f!, vec(initial_x), vec(initial_x)))
    ForwardDiff.checktag(jac_cfg, f!, vec(initial_x))
    
    fvec! = reshape_f(f!, initial_x)
    permf! = (fx::AbstractVector, x::AbstractVector) -> fvec!(x, fx)

    fx2 = vec(copy(initial_x))    
    function g!(x::AbstractVector, gx::AbstractMatrix)
        ForwardDiff.jacobian!(gx, permf!, fx2, x, jac_cfg, Val{false}())
    end

    function fg!(x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix)
        jac_res = DiffBase.DiffResult(fx, gx)
        ForwardDiff.jacobian!(jac_res, permf!, fx2, x, jac_cfg, Val{false}())
        DiffBase.value(jac_res)
    end

    return DifferentiableMultivariateFunction(fvec!, g!, fg!)
end
