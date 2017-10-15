function autodiff(f!, initial_x::AbstractArray)

    fvec! = reshape_f(f!, initial_x)
    permf! = (fx::AbstractVector, x::AbstractVector) -> fvec!(x, fx)

    fx2 = vec(copy(initial_x))
    jac_cfg = ForwardDiff.JacobianConfig(nothing, vec(initial_x), vec(initial_x))
    function g!(x::AbstractVector, gx::AbstractMatrix)
        ForwardDiff.jacobian!(gx, permf!, fx2, x, jac_cfg)
    end

    fg! = (x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix) -> begin
        jac_res = DiffBase.DiffResult(fx, gx)
        ForwardDiff.jacobian!(jac_res, permf!, fx2, x, jac_cfg)
        DiffBase.value(jac_res)
    end

    return DifferentiableMultivariateFunction(fvec!, g!, fg!)
end
