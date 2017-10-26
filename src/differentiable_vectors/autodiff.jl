# if only f! and vector shaped x_seed is given, construct j! or fj! according to autodiff
function DifferentiableVector(f!, x_seed::AbstractVector; J = nothing, autodiff=:central)
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
        return DifferentiableVector(f!, j!, fj!, x_seed; J = J)
    elseif autodiff == :forward
        return autodiff(f!, x_seed)
    end
end
function autodiff(f!, x_seed::AbstractArray)
    rf! = reshape_f(f!, x_seed)
    fx2 = vec(copy(x_seed))
    jac_cfg = ForwardDiff.JacobianConfig(nothing, vec(x_seed), vec(x_seed))
    function j!(jx, x)
        ForwardDiff.jacobian!(jx, rf!, fx2, x, jac_cfg)
    end

    fj! = (fx, jx, x) -> begin
        jac_res = DiffBase.DiffResult(fx, jx)
        ForwardDiff.jacobian!(jac_res, rf!, fx2, x, jac_cfg)
        DiffBase.value(jac_res)
    end

    return DifferentiableVector(rf!, j!, fj!, x_seed)
end
# if only f! and x_seed not as a vector is given, reshape f!, and construct
# j! and fj! according to autodiff differentiation
function DifferentiableVector(f!, x_seed::AbstractArray; autodiff=:central)
    return DifferentiableVector(reshape_f(f!, x_seed), x_seed; autodiff=autodiff)
end
