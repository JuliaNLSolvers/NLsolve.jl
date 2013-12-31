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
    local p_i
    try
        p_i = -J\r # Gauss-Newton step
    catch e
        if isa(e, Base.LinAlg.SingularException)
            # If Jacobian is singular, compute a least-squares solution to J*x+r=0
            U, S, V = svd(J)
            k = sum(S .> eps())
            mrinv = V*diagm([1./S[1:k]; zeros(length(S)-k)])*U' # Moore-Penrose generalized inverse of J
            p_i = -mrinv*r
        else
            throw(e)
        end
    end

    # Test is Gauss-Newton step is within the region
    if norm(p_i) <= delta
        copy!(p, p_i)
    else
        g = J'*r # Gradient direction
        p_c = - norm(g)^2/dot(g, (J'*J)*g)*g # Cauchy point

        if norm(p_c) >= delta
            # Cauchy point is out of the region, take the largest step along
            # gradient direction
            copy!(p, -delta/norm(g)*g)
        else
            # Compute the optimal point on dogleg path
            b = 2*dot(p_c, p_i-p_c)
            a = dot(p_i-p_c,p_i-p_c)
            tau = (-b+sqrt(b^2-4*a*(dot(p_c,p_c)-delta^2)))/2/a
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
                         extended_trace::Bool,
                         factor::Real)

    x = copy(initial_x)  # Current point
    xold = similar(x)    # Old point
    r = similar(x)       # Current residual
    r_new = similar(x)   # New residual
    p = similar(x)       # Step
    nn = length(x)
    J = Array(T, nn, nn) # Jacobian

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

    delta = NaN
    rho = NaN
    
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @trustregiontrace NaN

    delta = factor*norm(x)
    if delta == 0
        delta = factor
    end
    eta = 1e-4

    while !converged && it < iterations

        it += 1

        # Compute proposed iteration step
        dogleg!(p, r, J, delta)

        df.f!(x + p, r_new)
        f_calls += 1

        # Ratio of actual to predicted reduction
        rho = (norm(r)^2 - norm(r_new)^2)/(norm(r)^2 - norm(r+J*p)^2)

        copy!(xold, x)

        if rho > eta
            # Successful iteration
            x += p
            copy!(r, r_new)
            df.g!(x, J)
            g_calls += 1
            x_converged, f_converged, converged = assess_convergence(x, xold, r, xtol, ftol)
        else
            x_converged, converged = false, false
        end

        @trustregiontrace norm(x-xold)

        # Update size of trust region
        if rho < 0.25
            delta = 0.25*norm(p)
        elseif rho > 0.75 && abs(norm(p) - delta) < eps(delta)
            delta = 2*delta
        end
    end

    return SolverResults("Trust-region with dogleg",
                         initial_x, x, norm(r, Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         f_calls, g_calls)
end
