function autodiff(f!, initial_x::Vector)

    permf! = (fx, x) -> f!(x, fx)

    fx2 = copy(initial_x)
    g! = (x, gx) -> begin
        out = ForwardDiff.JacobianResult(fx2, gx)
        ForwardDiff.jacobian!(out, permf!, x)
    end

    fg! = (x, fx, gx) -> begin
        out = ForwardDiff.JacobianResult(fx, gx)
        ForwardDiff.jacobian!(out, permf!, x)
    end

    return DifferentiableMultivariateFunction(f!, g!, fg!)
end
