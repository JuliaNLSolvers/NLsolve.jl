struct NewtonTrustRegion
end

struct NewtonTrustRegionCache{Tx} <: AbstractSolverCache
    x::Tx
    xold::Tx
    r::Tx
    r_predict::Tx
    p::Tx
    p_c::Tx
    pi::Tx
    d::Tx
end
function NewtonTrustRegionCache(df)
    x = similar(df.x_f) # Current point
    xold = similar(x) # Old point
    r = similar(df.F)       # Current residual
    r_predict = similar(x)  # predicted residual
    p = similar(x)          # Step
    p_c = similar(x)          # Cauchy point
    pi = similar(x)          
    d = similar(x)          # Scaling vector
    NewtonTrustRegionCache(x, xold, r, r_predict, p, p_c, pi, d)
end
macro trustregiontrace(stepnorm)
    esc(quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(cache.x)
                dt["f(x)"] = copy(value(df))
                dt["g(x)"] = copy(jacobian(df))
                dt["delta"] = delta
                dt["rho"] = rho
            end
            update!(tr,
                    it,
                    maximum(abs, value(df)),
                    $stepnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end)
end

function dogleg!{T}(p::AbstractArray{T}, p_c::AbstractArray{T}, p_i, r::AbstractArray{T}, d::AbstractArray{T},
                    J::AbstractMatrix{T}, delta::Real)
    try
        copy!(p_i, J \ vec(r)) # Gauss-Newton step
    catch e
        if isa(e, Base.LinAlg.LAPACKException) || isa(e, Base.LinAlg.SingularException)
            # If Jacobian is singular, compute a least-squares solution to J*x+r=0
            U, S, V = svd(full(J)) # Convert to full matrix because sparse SVD not implemented as of Julia 0.3
            k = sum(S .> eps())
            mrinv = V * diagm([1./S[1:k]; zeros(eltype(S), length(S)-k)]) * U' # Moore-Penrose generalized inverse of J
            p_i .= mrinv * vec(r)
        else
            throw(e)
        end
    end
    scale!(p_i, -one(T))

    # Test if Gauss-Newton step is within the region
    if wnorm(d, p_i) <= delta
        copy!(p, p_i)   # accepts equation 4.13 from N&W for this step
    else
        # For intermediate we will use the output array `p` as a buffer to hold
        # the gradient. To make it easy to remember which variable that array
        # is representing we make g an alias to p and use g when we want the
        # gradient

        # compute g = J'r ./ (d .^ 2)
        g = p
        At_mul_B!(vec(g), J, vec(r))
        g .= g ./ d.^2

        # compute Cauchy point
        p_c .= -wnorm(d, g)^2 / sum(abs2, J*vec(g)) .* g

        if wnorm(d, p_c) >= delta
            # Cauchy point is out of the region, take the largest step along
            # gradient direction
            scale!(g, -delta/wnorm(d, g))

            # now we want to set p = g, but that is already true, so we're done

        else
            # from this point on we will only need p_i in the term p_i-p_c.
            # so we reuse the vector p_i by computing p_i = p_i - p_c and then
            # just so we aren't confused we name that p_diff
            p_i .-= p_c
            p_diff = p_i

            # Compute the optimal point on dogleg path
            b = 2 * wdot(d, p_c, d, p_diff)
            a = wnorm(d, p_diff)^2
            tau = (-b + sqrt(b^2 - 4a*(wnorm(d, p_c)^2 - delta^2)))/(2a)
            p_c .+= tau .* p_diff
            copy!(p, p_c)
        end
    end
end

function trust_region_{T}(df::OnceDifferentiable,
                          initial_x::AbstractArray{T},
                          xtol::T,
                          ftol::T,
                          iterations::Integer,
                          store_trace::Bool,
                          show_trace::Bool,
                          extended_trace::Bool,
                          factor::T,
                          autoscale::Bool,
                          cache = NewtonTrustRegionCache(df))
    copy!(cache.x, initial_x)
    value_jacobian!!(df, cache.x)
    cache.r .= value(df)
    check_isfinite(cache.r)
    
    it = 0
    x_converged, f_converged, converged = assess_convergence(value(df), ftol)
    delta = convert(T, NaN)
    rho = convert(T, NaN)
    if converged
        tr = SolverTrace()
        name = "Trust-region with dogleg"
        if autoscale
            name *= " and autoscaling"
        end
     
        return SolverResults(name,
        #initial_x, reshape(cache.x, size(initial_x)...), vecnorm(cache.r, Inf),
        initial_x, copy(cache.x), vecnorm(cache.r, Inf),
        it, x_converged, xtol, f_converged, ftol, tr,
        first(df.f_calls), first(df.df_calls))
    end
    
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @trustregiontrace convert(T, NaN)
    nn = length(cache.x)
    if autoscale
        for j = 1:nn
            cache.d[j] = norm(view(jacobian(df), :, j))
            if cache.d[j] == zero(T)
                cache.d[j] = one(T)
            end
        end
    else
        fill!(cache.d, one(T))
    end
    
    delta = factor * wnorm(cache.d, cache.x)
    if delta == zero(T)
        delta = factor
    end
    
    eta = convert(T, 1e-4)

    while !converged && it < iterations
        it += 1

        # Compute proposed iteration step
        dogleg!(cache.p, cache.p_c, cache.pi, cache.r, cache.d, jacobian(df), delta)

        copy!(cache.xold, cache.x)
        cache.x .+= cache.p
        value!(df, cache.x)

        # Ratio of actual to predicted reduction (equation 11.47 in N&W)
        A_mul_B!(vec(cache.r_predict), jacobian(df), vec(cache.p))
        cache.r_predict .+= cache.r
        rho = (sum(abs2, cache.r) - sum(abs2, value(df))) / (sum(abs2, cache.r) - sum(abs2, cache.r_predict))

        if rho > eta
            # Successful iteration
            cache.r .= value(df)
            jacobian!(df, cache.x)

            # Update scaling vector
            if autoscale
                for j = 1:nn
                    cache.d[j] = max(convert(T, 0.1) * cache.d[j], norm(view(jacobian(df), :, j)))
                end
            end

            x_converged, f_converged, converged = assess_convergence(cache.x, cache.xold, cache.r, xtol, ftol)
        else
            cache.x .-= cache.p
            x_converged, converged = false, false
        end

        @trustregiontrace euclidean(cache.x, cache.xold)

        # Update size of trust region
        if rho < 0.1
            delta = delta/2
        elseif rho >= 0.9
            delta = 2 * wnorm(cache.d, cache.p)
        elseif rho >= 0.5
            delta = max(delta, 2 * wnorm(cache.d, cache.p))
        end
    end

    name = "Trust-region with dogleg"
    if autoscale
        name *= " and autoscaling"
    end
    return SolverResults(name,
                         initial_x, copy(cache.x), vecnorm(cache.r, Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         first(df.f_calls), first(df.df_calls))
end

function trust_region{T}(df::OnceDifferentiable,
                         initial_x::AbstractArray{T},
                         xtol::Real,
                         ftol::Real,
                         iterations::Integer,
                         store_trace::Bool,
                         show_trace::Bool,
                         extended_trace::Bool,
                         factor::Real,
                         autoscale::Bool,
                         cache = NewtonTrustRegionCache(df))
    trust_region_(df, initial_x, convert(T,xtol), convert(T,ftol), iterations, store_trace, show_trace, extended_trace, convert(T,factor), autoscale)
end
