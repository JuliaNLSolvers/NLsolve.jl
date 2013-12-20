macro trustregiontrace(stepnorm)
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["f(x)"] = copy(r)
                dt["g(x)"] = copy(J)
                dt["delta"] = delta
                dt["rho"] = rho
            end
            update!(tr,
                    it,
                    maximum(abs(r)),
                    $stepnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end

function dogleg!{T}(p::Vector{T}, r::Vector{T}, J::Matrix{T}, delta::Real)
    g = J'*r
    if norm(g)^3/(delta*dot(g, (J'*J)*g)) >= 1
        copy!(p, -delta/norm(g)*g)
    else
        p_c = - norm(g)^2/dot(g, (J'*J)*g)*g
        ## FIXME: Handle singular J
        p_i = -J\r
            
        if norm(p_i) <= delta
            copy!(p, p_i)
        else
            b = 2*dot(p_c, p_i-p_c)
            a = dot(p_i-p_c,p_i-p_c)
            tau = (-b+sqrt(b^2-4*a*(dot(p_c,p_c)-delta^2)))/2*a
            copy!(p, p_c+tau*(p_i-p_c))
        end
    end
end
        
function trust_region{T}(df::DifferentiableMultivariateFunction,
                         initial_x::Vector{T},
                         xtol::Real,
                         ftol::Real,
                         iterations::Integer,
                         store_trace::Bool,
                         show_trace::Bool,
                         extended_trace::Bool)

    x = copy(initial_x)
    nn = length(x)
    xold = Array(T, nn)
    r = Array(T, nn)
    r_new = Array(T, nn)
    J = Array(T, nn, nn)
    p = Array(T, nn)

    # Count function calls
    f_calls, g_calls = 0, 0

    df.fg!(x, r, J)
    f_calls += 1
    g_calls += 1

    i = find(!isfinite(r))

    if !isempty(i)
        error("During the resolution of the non-linear system, the evaluation of the following equation(s) resulted in a non-finite number: $(i)")
    end

    it = 0
    x_converged, f_converged, converged = false, false, false

    # TODO: How should this flag be set?
    mayterminate = false

    delta = NaN
    rho = NaN
    
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @trustregiontrace NaN

    delta_max = 1. ## FIXME
    delta = 0.5 ## FIXME
    eta = 0.01 ## FIXME

    while !converged && it < iterations

        it += 1
        
        dogleg!(p, r, J, delta)

        df.f!(x + p, r_new)
        f_calls += 1
        
        rho = (norm(r)^2 - norm(r_new)^2)/(norm(r)^2 - norm(r+J*p)^2)

        copy!(xold, x)

        if rho > eta
            x += p
            copy!(r, r_new)
            df.g!(x, J)
            g_calls += 1
            x_converged, f_converged, converged = assess_convergence(x, xold, r, xtol, ftol)
        else
            x_converged, converged = false, false
        end

        @trustregiontrace norm(x-xold)

        if rho < 0.25
            delta = 0.25*norm(p)
        elseif rho > 0.75 && abs(norm(p) - delta) < eps(delta)
            delta = min(2*delta, delta_max)
        end

    end

    return SolverResults("Trust-region with dogleg",
                         initial_x, x, norm(r, Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         f_calls, g_calls)
end
