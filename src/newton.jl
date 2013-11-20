# Translation of Dynare's solve1.m

macro newtontrace(stepnorm)
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["f(x)"] = copy(fvec)
                dt["g(x)"] = copy(fjac)
            end
            update!(tr,
                    it,
                    maximum(abs(fvec)),
                    $stepnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end

function newton(df::DifferentiableMultivariateFunction,
                initial_x::Vector,
                xtol::Real,
                ftol::Real,
                iterations::Integer,
                store_trace::Bool,
                show_trace::Bool,
                extended_trace::Bool)

    x = copy(initial_x)
    nn = length(x)
    fvec = zeros(nn)
    fjac = zeros(nn,nn) 
    g = zeros(nn,1) 

    # Count function calls
    f_calls, g_calls = 0, 0

    tolmin = xtol 

    stpmx = 100 

    df.f!(x, fvec)
    f_calls += 1

    i = find(!isfinite(fvec))

    if !isempty(i)
        error("During the resolution of the non-linear system, the evaluation of the following equation(s) resulted in a non-finite number: $(i)")
    end

    f = 0.5*dot(fvec, fvec)

    stpmax = stpmx*max(norm(x, 2), nn) 
    first_time = 1

    it = 0
    x_converged, f_converged, converged = false, false, false

    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @newtontrace NaN

    while !converged && it < iterations

        it += 1
        
        df.fg!(x, fvec, fjac)
        f_calls += 1
        g_calls += 1

        g = fjac'*fvec

        p = -fjac\fvec

        xold = x 
        fold = f 

        (x,f,fvec,f_calls_update, g_calls_update)=lnsrch(xold,fold,g,p,stpmax,df,xtol)

        f_calls += f_calls_update
        g_calls += g_calls_update

        x_converged, f_converged, converged = assess_convergence(x, xold, fvec, xtol, ftol)

        @newtontrace norm(x-xold, 2)
    end

    return SolverResults("Newton with interpolating line-search",
                         initial_x, x, norm(fvec, Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         f_calls, g_calls)
end

function lnsrch(xold::Vector{Float64},fold::Float64,g::Vector{Float64},p::Vector{Float64},stpmax::Float64,df::DifferentiableMultivariateFunction,xtol::Real)
    local f, fvec
    
    alf = 1e-4 
    alam = 1

    # Count function calls
    f_calls, g_calls = 0, 0

    x = xold
    nn = length(x)
    fvec = zeros(nn)
    
    summ = sqrt(sum(p.*p))
    if !isfinite(summ)
        error("Some element of Newton direction isn't finite. Jacobian maybe singular or there is a problem with initial values")
    end

    if summ > stpmax
        p=p.*stpmax/summ
    end

    slope = dot(g,p)

    test = maximum(abs(p)'./maximum([abs(xold)';ones(1,nn)],1))
    alamin = xtol/test

    if alamin > 0.1
        alamin = 0.1
    end

    while true
        if alam < alamin
            break
        end
    
        x = xold + (alam*p)
        df.f!(x, fvec)
        f_calls += 1
        f = 0.5*dot(fvec,fvec)

        if any(isnan(fvec))
            alam = alam/2 
            alam2 = alam 
            f2 = f 
            fold2 = fold 
        else
            if f <= fold+alf*alam*slope
                break 
            else
                if alam == 1
                    tmplam = -slope/(2*(f-fold-slope)) 
                else
                    rhs1 = f-fold-alam*slope 
                    rhs2 = f2-fold2-alam2*slope 
                    a = (rhs1/(alam^2)-rhs2/(alam2^2))/(alam-alam2) 
                    b = (-alam2*rhs1/(alam^2)+alam*rhs2/(alam2^2))/(alam-alam2) 
                    if a == 0
                        tmplam = -slope/(2*b) 
                    else
                        disc = (b^2)-3*a*slope 

                        if disc < 0
                            error ("Roundoff problem")
                        else
                            tmplam = (-b+sqrt(disc))/(3*a) 
                        end

                    end

                    if tmplam > 0.5*alam
                        tmplam = 0.5*alam
                    end

                end

                alam2 = alam 
                f2 = f 
                fold2 = fold 
                alam = max(tmplam, 0.1*alam) 
            end
        end
    end
    return (x,f,fvec,f_calls,g_calls)
end
