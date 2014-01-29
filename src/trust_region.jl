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

function dogleg!{T}(p::Vector{T}, r::Vector{T}, d::Vector{T}, J::Matrix{T}, delta::Real)
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
    if norm(d .* p_i) <= delta
        copy!(p, p_i)
    else
        g = (J'*r) ./ (d .^ 2) # Gradient direction
        p_c = - norm(d .* g)^2/norm(J*g)^2*g # Cauchy point

        if norm(d .* p_c) >= delta
            # Cauchy point is out of the region, take the largest step along
            # gradient direction
            copy!(p, -delta/norm(d .* g)*g)
        else
            # Compute the optimal point on dogleg path
            b = 2*dot(d .* p_c, d .* (p_i-p_c))
            a = norm(d.*(p_i-p_c))^2
            tau = (-b+sqrt(b^2-4*a*(norm(d.*p_c)^2-delta^2)))/2/a
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
                         factor::Real,
                         autoscale::Bool)

    x = copy(initial_x)  # Current point
    nn = length(x)
    xold = nans(T, nn)   # Old point
    r = similar(x)       # Current residual
    r_new = similar(x)   # New residual
    p = similar(x)       # Step
    d = similar(x)       # Scaling vector
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
    x_converged, f_converged, converged = assess_convergence(x, xold, r, xtol, ftol)

    delta = NaN
    rho = NaN
    
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @trustregiontrace NaN

    if autoscale
        for j = 1:nn
            d[j] = norm(J[:,j])
            if d[j] == 0
                d[j] = 1
            end
        end
    else
        d = ones(nn)
    end
    delta = factor*norm(d .* x)
    if delta == 0
        delta = factor
    end
    eta = 1e-4

    while !converged && it < iterations

        it += 1

        # Compute proposed iteration step
        dogleg!(p, r, d, J, delta)

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

            # Update scaling vector
            if autoscale
                for j = 1:nn
                    d[j] = max(0.1*d[j], norm(J[:,j]))
                end
            end

            x_converged, f_converged, converged = assess_convergence(x, xold, r, xtol, ftol)
        else
            x_converged, converged = false, false
        end

        @trustregiontrace norm(x-xold)

        # Update size of trust region
        if rho < 0.1
            delta = 0.5*delta
        elseif rho >= 0.9
            delta = 2*norm(d .* p)
        elseif rho >= 0.5
            delta = max(delta, 2*norm(d .* p))
        end
    end

    name = "Trust-region with dogleg"
    if autoscale
        name *= " and autoscaling"
    end
    return SolverResults(name,
                         initial_x, x, norm(r, Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         f_calls, g_calls)
end
