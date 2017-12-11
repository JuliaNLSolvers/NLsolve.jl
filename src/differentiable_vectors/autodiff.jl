# if only f! and vector shaped x is given, construct j! or fj! according to autodiff
function OnceDifferentiable(f!, F::AbstractArray, x::AbstractArray, autodiff = :central)
    fv!(fxv::AbstractVector, xv::AbstractVector)  = f!(reshape(fxv, size(F)...), reshape(xv, size(x)...))
    OnceDifferentiable(fv!, vec(F), vec(x), autodiff)
end
function OnceDifferentiable(f!, F::AbstractVector, x::AbstractVector, autodiff = :central)
    if autodiff == :central
        function fj!(fx, jx, x)
            f!(fx, x)
            function f(x)
                fx = similar(x)
                f!(fx, x)
                return fx
            end
            finite_difference_jacobian!(f, x, fx, jx, autodiff)
        end
        function j!(jx, x)
            fx = similar(x)
            fj!(fx, jx, x)
        end
        return OnceDifferentiable(f!, j!, fj!, similar(x), similar(x))
    elseif autodiff == :forward || autodiff == true
        jac_cfg = ForwardDiff.JacobianConfig(f!, x, x)
        ForwardDiff.checktag(jac_cfg, f!, x)
    
        fx2 = copy(x)   
        function g!(gx, x)
            ForwardDiff.jacobian!(gx, f!, fx2, x, jac_cfg, Val{false}())
        end
        function fg!(fx, gx, x)
            jac_res = DiffBase.DiffResult(fx, gx)
            ForwardDiff.jacobian!(jac_res, f!, fx2, x, jac_cfg, Val{false}())
            DiffBase.value(jac_res)
        end
    
        return OnceDifferentiable(f!, g!, fg!, similar(x), x)
    else
        error("The autodiff value $(autodiff) is not supported. Use :central or :forward.")
    end
end
