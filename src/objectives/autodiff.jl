function OnceDifferentiable(f!, x::AbstractArray, F::AbstractArray, autodiff::Union{Symbol, Bool} = :central)
    if autodiff == :central
        central_cache = DiffEqDiffTools.JacobianCache(similar(x), similar(x), similar(x))
        function fj!(F, J, x)
            f!(F, x)
            DiffEqDiffTools.finite_difference_jacobian!(J, f!, x, central_cache)
            F
        end
        function j!(J, x)
            F = similar(x)
            fj!(F, J, x)
        end
        return OnceDifferentiable(f!, j!, fj!, x, x)
    elseif autodiff == :forward || autodiff == true
        jac_cfg = ForwardDiff.JacobianConfig(f!, x, x)
        ForwardDiff.checktag(jac_cfg, f!, x)
    
        F2 = copy(x)   
        function g!(J, x)
            ForwardDiff.jacobian!(J, f!, F2, x, jac_cfg, Val{false}())
        end
        function fg!(F, J, x)
            jac_res = DiffBase.DiffResult(F, J)
            ForwardDiff.jacobian!(jac_res, f!, F2, x, jac_cfg, Val{false}())
            DiffBase.value(jac_res)
        end
    
        return OnceDifferentiable(f!, g!, fg!, x, x)
    else
        error("The autodiff value $(autodiff) is not supported. Use :central or :forward.")
    end
end
