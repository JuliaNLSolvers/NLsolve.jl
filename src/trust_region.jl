macro trustregiontrace(stepnorm)
    esc(quote
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
    end)
end

function dogleg!{T}(p::Vector{T}, r::Vector{T}, d2::Vector{T}, J::AbstractMatrix{T}, delta::Real)
    local p_i
    try
        p_i = J \ r # Gauss-Newton step
    catch e
        if isa(e, Base.LinAlg.LAPACKException) || isa(e, Base.LinAlg.SingularException)
            # If Jacobian is singular, compute a least-squares solution to J*x+r=0
            U, S, V = svd(full(J)) # Convert to full matrix because sparse SVD not implemented as of Julia 0.3
            k = sum(S .> eps())
            mrinv = V * diagm([1./S[1:k]; zeros(eltype(S), length(S)-k)]) * U' # Moore-Penrose generalized inverse of J
            p_i = mrinv * r
        else
            throw(e)
        end
    end
    scale!(p_i, -one(T))
    # Test if Gauss-Newton step is within the region
    if wnorm(d2, p_i) <= delta
        copy!(p, p_i)
    else
        p_c = At_mul_B(J, r) # Gradient direction
        broadcast!(/, p_c, p_c, d2)
        scale!(p_c, - wnorm(d2, p_c)^2 / sum(abs2,J * p_c)) # Cauchy point
        if wnorm(d2, p_c) >= delta
            # Cauchy point is out of the region, take the largest step along
            # gradient direction
            scale!(p, p_c, delta / wnorm(d2, p_c))
        else
            p_diff = Base.BLAS.axpy!(-one(T), p_c, p_i)
            # Compute the optimal point on dogleg path
            b = 2 * wdot(d2, p_c, p_diff)
            a = wdot(d2, p_diff, p_diff)
            c = wdot(d2, p_c, p_c)
            tau = (-b + sqrt(b^2 - 4 * a * (c - delta^2)))/(2*a)
            copy!(p, p_c)
            Base.BLAS.axpy!(tau, p_diff, p)
        end
    end
end

function trust_region_{T}(df::AbstractDifferentiableMultivariateFunction,
                         initial_x::Vector{T},
                         xtol::T,
                         ftol::T,
                         iterations::Integer,
                         store_trace::Bool,
                         show_trace::Bool,
                         extended_trace::Bool,
                         factor::T,
                         autoscale::Bool)

    x = copy(initial_x)     # Current point
    nn = length(x)
    xold = fill(convert(T, NaN), nn) # Old point
    r = similar(x)          # Current residual
    r_new = similar(x)      # New residual
    r_predict = similar(x)  # predicted residual
    p = similar(x)          # Step
    d2 = similar(x)         # Scaling vector
    J = alloc_jacobian(df, T, nn)    # Jacobian

    # Count function calls
    f_calls, g_calls = 0, 0

    df.fg!(x, r, J)
    f_calls += 1
    g_calls += 1

    check_isfinite(r)

    it = 0
    x_converged, f_converged, converged = assess_convergence(x, xold, r, xtol, ftol)

    delta = convert(T, NaN)
    rho = convert(T, NaN)

    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @trustregiontrace convert(T, NaN)

    if autoscale
        for j = 1:nn
            d2[j] = sum(abs2,view(J, :, j))
            if d2[j] == zero(T)
                d2[j] = one(T)
            end
        end
    else
        fill!(d2, one(T))
    end
    delta = factor * wnorm(d2, x)
    if delta == 0
        delta = factor
    end
    eta = convert(T, 1e-4)

    while !converged && it < iterations

        it += 1

        # Compute proposed iteration step
        dogleg!(p, r, d2, J, delta)

        copy!(xold, x)
        Base.BLAS.axpy!(one(T), p, x)
        df.f!(x, r_new)
        f_calls += 1

        # Ratio of actual to predicted reduction
        A_mul_B!(r_predict, J, p)
        Base.BLAS.axpy!(one(T), r, r_predict)
        rho = (sum(abs2,r) - sum(abs2,r_new)) / (sum(abs2,r) - sum(abs2,r_predict))

        if rho > eta
            # Successful iteration
            copy!(r, r_new)
            df.g!(x, J)
            g_calls += 1

            # Update scaling vector
            if autoscale
                for j = 1:nn
                    d2[j] = max(convert(T, 0.01) * d2[j], sum(abs2,view(J, :, j)))
                end
            end

            x_converged, f_converged, converged = assess_convergence(x, xold, r, xtol, ftol)
        else
            Base.BLAS.axpy!(-one(T), p, x)
            x_converged, converged = false, false
        end

        @trustregiontrace sqeuclidean(x, xold)

        # Update size of trust region
        if rho < 0.1
            delta = delta/2
        elseif rho >= 0.9
            delta = 2 * wnorm(d2, p)
        elseif rho >= 0.5
            delta = max(delta, 2 * wnorm(d2, p))
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

function trust_region{T}(df::AbstractDifferentiableMultivariateFunction,
                         initial_x::Vector{T},
                         xtol::Real,
                         ftol::Real,
                         iterations::Integer,
                         store_trace::Bool,
                         show_trace::Bool,
                         extended_trace::Bool,
                         factor::Real,
                         autoscale::Bool)
    trust_region_(df, initial_x, convert(T,xtol), convert(T,ftol), iterations, store_trace, show_trace, extended_trace, convert(T,factor), autoscale)
end
